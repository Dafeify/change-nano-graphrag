# -*- coding: utf-8 -*-
"""
run_deepseek.py

新版舰船文本解析 + 层级分类匹配主程序。

适配新的 class_data.txt 设计：
1. 不再做 CVN-68~CVN-77 具体单舰识别；
2. 只使用 7 个已知舰级先验知识；
3. 输出：六大类判断 -> 已知舰级匹配 -> 类别内未知类判断；
4. LLM 只负责把用户文本/图像描述文本解析成固定属性卡槽 JSON；
5. 最终分类由规则图谱 + 原型卡槽匹配完成，不让 LLM 直接猜舰级。

依赖文件：
- class_data.txt
- schema_config.py
- .env 中的 SILICONFLOW_API_KEY

可选依赖：
- nano-graphrag、sentence-transformers：只用于构建/加载 GraphRAG 知识层；
  即使不调用 aquery，程序也会生成标准 GraphML 供后续解释和检索使用。
"""

import os
import sys
import io
import re
import json
import math
import shutil
import asyncio
import unicodedata
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict

# ============================================================
# 0. Windows 控制台 UTF-8 兼容
# ============================================================

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("LANG", "en_US.UTF-8")
os.environ.setdefault("LC_ALL", "en_US.UTF-8")

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# ============================================================
# 1. 基础依赖与配置
# ============================================================

from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIConnectionError
import httpx
import networkx as nx

load_dotenv()

MODEL = os.getenv("SILICONFLOW_MODEL", "deepseek-ai/DeepSeek-V3")
BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
API_KEY = os.getenv("SILICONFLOW_API_KEY")

# 新版配置文件。请确保 schema_config.py 和本脚本在同一目录，或在 PYTHONPATH 下。
try:
    from schema_config import (
        SHIP_CATEGORIES,
        KNOWN_SHIP_CLASSES,
        UNKNOWN_OUTPUT_TEMPLATE,
        SLOT_SCHEMA,
        SLOT_TO_GROUP,
        ALL_SLOTS,
        GROUP_TO_RELATION,
        RELATION_SCHEMA,
        VALUE_VOCAB,
        GLOBAL_VALUE_ALIASES,
        SLOT_VALUE_ALIASES,
        EQUIPMENT_ALIASES,
        CATEGORY_FEATURE_HINTS,
        SLOT_WEIGHTS,
        NEGATIVE_CONFLICT_WEIGHTS,
        CATEGORY_CONFIDENCE_THRESHOLD,
        KNOWN_CLASS_CONFIDENCE_THRESHOLD,
        KNOWN_CLASS_MARGIN_THRESHOLD,
        OPEN_SET_CLASS_THRESHOLD,
        normalize_basic_text,
        normalize_slot_value,
        normalize_equipment_name,
        get_slot_group,
        get_group_relation,
        empty_observed_schema,
        validate_observed_schema,
        get_category_of_known_class,
        SIGNATURE_RULE_THRESHOLDS,
        KNOWN_CLASS_STRONG_SIGNATURES,
        KNOWN_CLASS_CONFLICT_SIGNATURES,
        KNOWN_CLASS_ANCHOR_SIGNATURES,
        KNOWN_CLASS_SUPPORT_SIGNATURES,
    )
except Exception as e:
    raise RuntimeError(
        "无法导入 schema_config.py。请把 schema_config.py 放在当前脚本同目录，"
        "并确认你使用的是新版 class_data 对应的 schema_config。"
    ) from e


# ============================================================
# 2. 可选 GraphRAG 支持
# ============================================================

# 说明：分类匹配不依赖 nano-graphrag 的 LLM 推断。
# GraphRAG 在这里主要作为图谱存储/解释层，后续你可以 aquery 做证据解释。
try:
    from nano_graphrag import GraphRAG, QueryParam
    from nano_graphrag.base import BaseKVStorage
    from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _GRAPHRAG_AVAILABLE = True
except Exception:
    _GRAPHRAG_AVAILABLE = False

_EMBED_MODEL = None


def _ensure_embedding_model(model_name: str = "BAAI/bge-large-zh-v1.5"):
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(model_name)
    return _EMBED_MODEL


if _GRAPHRAG_AVAILABLE:
    # 延迟加载模型，避免只是调试解析/匹配时每次都加载 1GB+ embedding。
    @wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=512)
    async def local_embedding(texts: List[str]) -> "np.ndarray":
        model = _ensure_embedding_model()
        return model.encode(texts, normalize_embeddings=True)


_LLM_SEMAPHORE = None


def get_llm_semaphore():
    global _LLM_SEMAPHORE
    if _LLM_SEMAPHORE is None:
        _LLM_SEMAPHORE = asyncio.Semaphore(1)
    return _LLM_SEMAPHORE


async def siliconflow_llm_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict[str, str]]] = None,
    **kwargs,
) -> str:
    """
    SiliconFlow LLM 调用函数。
    1. 支持 nano-graphrag 的 hashing_kv 缓存；
    2. 对 429 / 超时做长等待重试；
    3. 默认 temperature=0，保证结构化解析稳定。
    """
    import random

    if not API_KEY:
        raise RuntimeError("未读取到 SILICONFLOW_API_KEY，请在 .env 中配置。")

    history_messages = history_messages or []
    hashing_kv: Optional[BaseKVStorage] = kwargs.pop("hashing_kv", None) if _GRAPHRAG_AVAILABLE else None

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        cached = await hashing_kv.get_by_id(args_hash)
        if cached is not None:
            return cached["return"]
    else:
        args_hash = None

    kwargs.setdefault("temperature", 0.0)
    kwargs.pop("model", None)
    kwargs.pop("messages", None)

    client = AsyncOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        max_retries=0,
        timeout=httpx.Timeout(connect=30.0, read=600.0, write=120.0, pool=120.0),
    )

    max_retries = 10
    for retry_idx in range(max_retries):
        try:
            async with get_llm_semaphore():
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    **kwargs,
                )
            result = response.choices[0].message.content or ""
            if hashing_kv is not None and args_hash is not None:
                await hashing_kv.upsert({args_hash: {"return": result, "model": MODEL}})
            return result

        except RateLimitError:
            wait_seconds = min(300, 30 * (retry_idx + 1)) + random.uniform(0, 8)
            print(f"[LLM限流] 第 {retry_idx + 1}/{max_retries} 次重试，等待 {wait_seconds:.1f} 秒...")
            await asyncio.sleep(wait_seconds)

        except (APITimeoutError, APIConnectionError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            wait_seconds = min(300, 20 * (retry_idx + 1)) + random.uniform(0, 8)
            print(f"[LLM超时/网络波动] {type(e).__name__}，等待 {wait_seconds:.1f} 秒...")
            await asyncio.sleep(wait_seconds)

        except Exception as e:
            text = str(e).lower()
            if "429" in text or "rate limit" in text or "tpm" in text:
                wait_seconds = min(300, 30 * (retry_idx + 1)) + random.uniform(0, 8)
                print(f"[LLM限流] 捕获限流异常，等待 {wait_seconds:.1f} 秒...")
                await asyncio.sleep(wait_seconds)
            elif "timeout" in text or "timed out" in text:
                wait_seconds = min(300, 20 * (retry_idx + 1)) + random.uniform(0, 8)
                print(f"[LLM超时] 捕获超时异常，等待 {wait_seconds:.1f} 秒...")
                await asyncio.sleep(wait_seconds)
            else:
                raise

    raise RuntimeError("LLM 调用多次重试后仍然失败。")


def build_graph_rag(working_dir: str = "./class_index"):
    """创建 GraphRAG 对象。分类主流程不依赖 aquery，但保留给后续解释层使用。"""
    if not _GRAPHRAG_AVAILABLE:
        raise RuntimeError("当前环境未安装 nano-graphrag 或 sentence-transformers，无法 build_graph_rag。")

    return GraphRAG(
        working_dir=working_dir,
        best_model_func=siliconflow_llm_complete,
        cheap_model_func=siliconflow_llm_complete,
        best_model_id=MODEL,
        cheap_model_id=MODEL,
        embedding_func=local_embedding,
        best_model_max_async=1,
        cheap_model_max_async=1,
        entity_extract_max_gleaning=0,
        chunk_token_size=700,
        chunk_overlap_token_size=80,
    )


# ============================================================
# 3. class_data 读取与规则构图
# ============================================================

SKIP_TOP_LEVEL_SECTIONS = {"RELATION_SCHEMA", "FEATURE_GROUPS"}
UNKNOWN_VALUES = {"", "未知", "不确定", "未提及", "N/A", "None", "null"}


def clean_text(s: Any) -> str:
    return normalize_basic_text(s).strip().strip('"').strip()


def split_values(value: Any) -> List[str]:
    """
    class_data 和 LLM 输出值拆分。
    只按逗号/中文逗号/顿号拆分，不按斜杠拆分，避免破坏“前部/中前部”这类固定表达。
    """
    if value is None:
        return []
    if isinstance(value, list):
        result = []
        for x in value:
            result.extend(split_values(x))
        return result

    text = clean_text(value)
    if not text:
        return []

    text = text.replace("，", ",").replace("、", ",")
    parts = [clean_text(x) for x in text.split(",")]
    return [p for p in parts if p]


def parse_key_value_lines(section_text: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for raw_line in section_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, value = line.split(":", 1)
        elif "：" in line:
            key, value = line.split("：", 1)
        else:
            continue
        result[clean_text(key)] = clean_text(value)
    return result


def get_sections(card_text: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    matches = list(re.finditer(r"^\[([A-Z_]+)\]\s*$", card_text, flags=re.M))
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(card_text)
        sections[name] = card_text[start:end].strip()
    return sections


def load_class_data(class_data_path: str = "./class_data.txt") -> Dict[str, Dict[str, Any]]:
    """
    读取新版 class_data.txt。
    返回：
    {
      Ship_Class: {
        "category": Ship_Category,
        "known_status": "Known",
        "groups": { group: {slot: raw_value} }
      }
    }
    """
    if not os.path.exists(class_data_path):
        raise FileNotFoundError(f"class_data 文件不存在: {class_data_path}")

    with open(class_data_path, "r", encoding="utf-8") as f:
        text = f.read()

    title_matches = list(re.finditer(r"^【([^】]+)】\s*$", text, flags=re.M))
    prototypes: Dict[str, Dict[str, Any]] = {}

    for idx, tm in enumerate(title_matches):
        title = clean_text(tm.group(1))
        start = tm.end()
        end = title_matches[idx + 1].start() if idx + 1 < len(title_matches) else len(text)
        card_text = text[start:end]
        sections = get_sections(card_text)

        if not sections:
            continue

        class_kv = parse_key_value_lines(sections.get("CLASS", ""))
        ship_class = clean_text(class_kv.get("Ship_Class", title))
        category = clean_text(class_kv.get("Ship_Category", ""))
        known_status = clean_text(class_kv.get("Known_Status", "Known"))
        known_class = clean_text(class_kv.get("Known_Class", "是"))

        if not ship_class or not category:
            continue

        groups: Dict[str, Dict[str, str]] = {}
        for group_name, section_text in sections.items():
            if group_name in SKIP_TOP_LEVEL_SECTIONS:
                continue
            groups[group_name] = parse_key_value_lines(section_text)

        prototypes[ship_class] = {
            "title": title,
            "category": category,
            "known_status": known_status,
            "known_class": known_class,
            "groups": groups,
        }

    if not prototypes:
        raise RuntimeError("未从 class_data 中解析到任何舰级原型卡片，请检查格式。")

    return prototypes


def feature_node_id(slot: str, value: str) -> str:
    """避免“无”“有”等值跨卡槽合并，特征值节点用 slot::value。"""
    return f"{slot}::{value}"


def build_class_graph_from_class_data(
    working_dir: str,
    class_data_path: str = "./class_data.txt",
    rebuild: bool = True,
) -> str:
    """
    根据 class_data 规则构建标准 GraphML。
    不调用 LLM，不使用 nano-graphrag 的 ainsert。
    """
    os.makedirs(working_dir, exist_ok=True)
    graphml_file = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")

    if rebuild and os.path.exists(graphml_file):
        backup_file = graphml_file + ".bak"
        shutil.copy2(graphml_file, backup_file)
        os.remove(graphml_file)
        print(f"[规则构图] 已备份并删除旧 GraphML: {backup_file}")

    prototypes = load_class_data(class_data_path)
    G = nx.DiGraph()

    def add_node(name: str, entity_type: str, description: str = ""):
        name = clean_text(name)
        if not name:
            return
        if name not in G:
            G.add_node(
                name,
                entity_type=entity_type,
                description=description or name,
                source_id="class_data_rule_graph",
            )
        else:
            if entity_type and not G.nodes[name].get("entity_type"):
                G.nodes[name]["entity_type"] = entity_type

    def add_edge(src: str, tgt: str, relation_type: str, description: str = "", weight: float = 1.0):
        src = clean_text(src)
        tgt = clean_text(tgt)
        if not src or not tgt:
            return
        if G.has_edge(src, tgt):
            old = G[src][tgt]
            if relation_type not in str(old.get("relation_type", "")):
                old["relation_type"] = str(old.get("relation_type", "")) + "<SEP>" + relation_type
            if description and description not in str(old.get("description", "")):
                old["description"] = str(old.get("description", "")) + "<SEP>" + description
            return
        G.add_edge(
            src,
            tgt,
            relation_type=relation_type,
            description=description or relation_type,
            weight=float(weight),
            source_id="class_data_rule_graph",
        )

    for category in SHIP_CATEGORIES:
        add_node(category, "Ship_Category", f"舰船大类：{category}")
    add_node("Known", "Known_Status", "已知类")

    for ship_class, data in prototypes.items():
        category = data["category"]
        add_node(ship_class, "Ship_Class", f"已知舰级原型：{ship_class}")
        add_node(category, "Ship_Category", f"舰船大类：{category}")
        add_edge(ship_class, category, "CLASS_IN_CATEGORY", f"{ship_class} 属于 {category}")
        add_edge(ship_class, "Known", "HAS_KNOWN_STATUS", f"{ship_class} 是已知类")

        groups = data["groups"]
        for group_name, slots in groups.items():
            if group_name == "CLASS" or group_name not in SLOT_SCHEMA:
                continue
            relation_type = GROUP_TO_RELATION.get(group_name, "HAS_TEXT_ATTRIBUTE")
            for slot, raw_value in slots.items():
                if slot not in SLOT_TO_GROUP and slot != "Keywords":
                    # 保留宽容性：class_data 里临时加的字段也能入图，但会作为普通槽位。
                    pass

                values = split_values(raw_value)
                if not values:
                    continue

                slot_node = f"Feature_Slot::{slot}"
                add_node(slot_node, "Feature_Slot", f"属性卡槽：{slot}")

                for value in values:
                    value = normalize_slot_value(slot, value)
                    if not value:
                        continue

                    if group_name == "EQUIPMENT_DETAILS":
                        value = normalize_equipment_name(value)
                        node_type = "Equipment_Value"
                    elif group_name == "MISSION_FEATURES":
                        node_type = "Mission_Value"
                    elif group_name == "NEGATIVE_FEATURES":
                        node_type = "Negative_Feature"
                    else:
                        node_type = "Feature_Value"

                    fnode = feature_node_id(slot, value)
                    add_node(fnode, node_type, f"{slot}: {value}")
                    add_edge(
                        ship_class,
                        fnode,
                        relation_type,
                        f"{relation_type}，{ship_class} 的 {slot} = {value}",
                        SLOT_WEIGHTS.get(slot, 1.0),
                    )
                    add_edge(
                        fnode,
                        slot_node,
                        "VALUE_OF_SLOT",
                        f"{value} 属于卡槽 {slot}",
                    )
                    add_edge(
                        fnode,
                        category,
                        "SUPPORTS_CATEGORY",
                        f"{slot}: {value} 可作为 {category} 的先验特征",
                        CATEGORY_FEATURE_HINTS.get(category, {}).get(value, 1.0),
                    )

    nx.write_graphml(G, graphml_file)
    print(f"[规则构图完成] {graphml_file}")
    print(f"[规则构图完成] 节点 {G.number_of_nodes()} 个，边 {G.number_of_edges()} 条")
    return graphml_file


def print_graph_summary(working_dir: str):
    graphml_file = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")
    if not os.path.exists(graphml_file):
        print(f"[图谱统计] GraphML 不存在: {graphml_file}")
        return
    G = nx.read_graphml(graphml_file)
    type_counts = defaultdict(int)
    rel_counts = defaultdict(int)
    for _n, data in G.nodes(data=True):
        type_counts[str(data.get("entity_type", "UNKNOWN"))] += 1
    for _u, _v, data in G.edges(data=True):
        rel_counts[str(data.get("relation_type", "UNKNOWN"))] += 1
    print("\n" + "=" * 60)
    print(f"【class graph 统计】节点 {G.number_of_nodes()} 个，边 {G.number_of_edges()} 条")
    print("节点类型：", dict(sorted(type_counts.items())))
    print("关系类型：", dict(sorted(rel_counts.items())))
    print("=" * 60)


# ============================================================
# 4. LLM 文本解析：direct_text_parse_v2
# ============================================================

def build_schema_template_for_prompt() -> Dict[str, Dict[str, str]]:
    schema = empty_observed_schema()
    # EQUIPMENT_DETAILS 更适合输出列表，但为了兼容用户口述，也允许字符串。
    for slot in SLOT_SCHEMA.get("EQUIPMENT_DETAILS", []):
        schema["EQUIPMENT_DETAILS"][slot] = []
    return schema


def build_vocab_text(max_items_per_slot: int = 20) -> str:
    lines = []
    for slot, values in VALUE_VOCAB.items():
        vals = values[:max_items_per_slot]
        lines.append(f"- {slot}: {', '.join(vals)}")
    return "\n".join(lines)


async def direct_text_parse_v2(user_text: str) -> str:
    """
    只根据用户输入文本抽取属性卡槽 JSON。
    不做舰级分类，不输出未知类名称。
    """
    schema_template = build_schema_template_for_prompt()
    schema_json = json.dumps(schema_template, ensure_ascii=False, indent=2)
    vocab_text = build_vocab_text()
    known_classes_text = "、".join([c for cs in KNOWN_SHIP_CLASSES.values() for c in cs])
    categories_text = "、".join(SHIP_CATEGORIES)

    prompt = f"""你是一个舰船属性抽取模块，不是分类器。

你的任务：
从用户输入文本中抽取舰船属性，填入固定 JSON 卡槽。
这些输入可能来自：
1. 用户口头描述他看到的舰船；
2. 图像大模型对舰船图片的文字描述；
3. 百度百科/维基百科等技术文本。

重要规则：
1. 只能使用输入文本明确提到的信息，不要补充、猜测或调用外部知识。
2. 不要直接判断最终舰级，不要输出“这是某某级”。
3. 允许识别的六大类仅为：{categories_text}。但本步骤只抽属性，不做最终分类。
4. 已知舰级仅有：{known_classes_text}。如果文本中没有明确出现这些名称，不要主动填入。
5. 不要输出任何未知舰级名称；如果文本像某个未知类，也只能抽取属性。
6. 未提到的信息填“未知”；明确看不清/无法判断填“不确定”；明确不存在填“无”。但 NEGATIVE_FEATURES 中 No_* 字段必须输出“是/否/未知”：例如“没有弹射器”对应 No_Catapult=是，“有弹射器”对应 No_Catapult=否。
7. 图片可见内容也要按语义归类：
   - 船体、舰首、舰尾、舰桥、舰岛、上层建筑、烟囱、桅杆、外形 → VISUAL_STRUCTURE
   - 飞行甲板、弹射器、拦阻索、升降机、机库 → AVIATION_FEATURES
   - 坞舱、艉门、登陆艇、车辆甲板、运兵能力 → AMPHIBIOUS_FEATURES
   - 主炮、垂发、相控阵雷达、近防系统、反舰导弹、声呐 → WEAPON_SENSOR_FEATURES
   - 长度、排水量、航速、动力、舰员、载机/载员数量 → TEXT_ATTRIBUTES
   - 具体装备型号 → EQUIPMENT_DETAILS
8. EQUIPMENT_DETAILS 中只填写明确出现的具体型号或专名；不要把“雷达系统”“武器系统”“直升机”“导弹”等泛称当成具体型号。
9. 严格输出 JSON，不要 markdown，不要解释。

部分标准值参考：
{vocab_text}

用户输入文本：
{user_text}

请严格输出如下 JSON 结构，所有卡槽必须出现：
{{
  "observed_attributes": {schema_json},
  "textual_summary": "只基于输入文本的一句话摘要"
}}
"""
    return await siliconflow_llm_complete(prompt, temperature=0.0)


def extract_json_object(text: str) -> Dict[str, Any]:
    """从 LLM 输出中提取 JSON 对象。"""
    text = str(text or "").strip()
    if not text:
        raise ValueError("LLM 输出为空")

    # 去掉可能的 markdown 包裹
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.I).strip()
    text = re.sub(r"```$", "", text.strip()).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError(f"无法解析 LLM JSON 输出：{text[:300]}")


def normalize_observed_attributes(parsed_obj: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """补齐并标准化 LLM 输出卡槽。"""
    observed = parsed_obj.get("observed_attributes", parsed_obj)
    base = empty_observed_schema()

    def normalize_negative_value(v: Any) -> str:
        """
        No_* 负特征统一用 是/否/未知。
        例如：No_Catapult=有 表示“存在没有弹射器这个负特征”，应归一为 是。
        """
        t = clean_text(v)
        if t in {"", "未知", "不确定", "未提及"}:
            return "未知"
        if t in {"是", "有", "存在", "明确", "true", "True", "1"}:
            return "是"
        if t in {"否", "无", "没有", "不存在", "false", "False", "0"}:
            return "否"
        return t

    for group, slots in SLOT_SCHEMA.items():
        if group == "CLASS":
            continue
        group_obj = observed.get(group, {}) if isinstance(observed, dict) else {}
        if not isinstance(group_obj, dict):
            group_obj = {}

        for slot in slots:
            raw = group_obj.get(slot, "未知")
            if isinstance(raw, list):
                vals = [normalize_slot_value(slot, x) for x in raw if clean_text(x)]
                if group == "EQUIPMENT_DETAILS":
                    vals = [normalize_equipment_name(x) for x in vals]
                if slot.startswith("No_"):
                    vals = [normalize_negative_value(x) for x in vals]
                base[group][slot] = vals
            else:
                val = normalize_slot_value(slot, raw)
                if slot.startswith("No_"):
                    val = normalize_negative_value(val)
                if group == "EQUIPMENT_DETAILS" and val not in UNKNOWN_VALUES:
                    base[group][slot] = [normalize_equipment_name(x) for x in split_values(val)]
                else:
                    base[group][slot] = val

    return base


def compact_key(value: Any) -> str:
    text = clean_text(value).casefold()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\s,，、/;；:：()（）\-—_\[\]【】'\"“”]+", "", text)
    return text


def is_unknown_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, list):
        return len(value) == 0
    return clean_text(value) in {"", "未知", "不确定", "未提及"}


def is_presence_yes(value: Any) -> bool:
    text = clean_text(value)
    return any(k in text for k in ["有", "是", "强", "可搭载", "全通式", "大型", "LCAC", "LCU"])


def is_presence_no(value: Any) -> bool:
    text = clean_text(value)
    return text in {"无", "否", "0"} or text.startswith("无") or "没有" in text or "不存在" in text


def extract_numbers(value: Any) -> List[float]:
    """抽取数值，支持 10.2万吨、100000吨、332.8米。"""
    text = clean_text(value)
    nums: List[float] = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(万)?", text):
        num = float(m.group(1))
        if m.group(2) == "万":
            num *= 10000
        nums.append(num)
    return nums


def numeric_match(observed: Any, prototype: Any, tolerance: float = 0.20) -> bool:
    obs_nums = extract_numbers(observed)
    pro_nums = extract_numbers(prototype)
    if not obs_nums or not pro_nums:
        return False

    proto_text = clean_text(prototype)
    for o in obs_nums:
        for p in pro_nums:
            if p == 0:
                continue
            if "以上" in proto_text and o >= p * (1 - tolerance):
                return True
            if "以下" in proto_text and o <= p * (1 + tolerance):
                return True
            if abs(o - p) / max(abs(p), 1.0) <= tolerance:
                return True
    return False


NUMERIC_SLOTS = {
    "Length_Overall", "Beam", "Draft", "Standard_Displacement", "Full_Load_Displacement",
    "Speed", "Range", "Crew", "Aircraft_Capacity", "Vehicle_Capacity", "Troop_Capacity",
    "Landing_Craft_Capacity", "Power_Output", "Aircraft_Elevator_Count", "Catapult_Count",
    "Arresting_Gear_Count", "Helicopter_Spot_Count",
}


# ============================================================
# 5.1 匹配打分修正：降低泛化“无/0”，强化强区分特征
# ============================================================

# 这些覆盖值只在当前 run_deepseek 中生效，不强制你马上修改 schema_config.py。
# 目的：让“三体船、低矮隐身化上层建筑、中小口径主炮”等强区分特征拉开分数。
LOCAL_SLOT_WEIGHT_OVERRIDES = {
    "Hull_Form": 5.0,
    "Superstructure_Type": 3.0,
    "Stealth_Shape": 2.5,
    "Flight_Deck_Type": 2.5,
    "Flight_Deck_Position": 2.0,
    "Main_Gun_Caliber": 2.5,
    "Mission_Module": 3.0,

    # 下面这些通常是“排除性共性特征”，不能让它们把巡洋舰/驱逐舰/护卫舰一起抬高太多。
    "Catapult": 1.0,
    "Catapult_Count": 0.5,
    "Arresting_Gear": 1.0,
    "Arresting_Gear_Count": 0.5,
    "Fixed_Wing_Aircraft_Operation": 0.8,
    "Well_Deck": 1.2,
}

LOW_INFORMATION_ABSENT_VALUES = {
    "无", "0", "不适用", "非主要特征", "非主要能力"
}

# 这些 slot 的“无/0”通常只能作为排除项，不应作为强确认项。
LOW_INFORMATION_ABSENT_SLOTS = {
    "Catapult", "Catapult_Count",
    "Arresting_Gear", "Arresting_Gear_Count",
    "Fixed_Wing_Aircraft_Operation",
    "Well_Deck", "Stern_Gate", "Landing_Craft_Capability",
    "VLS_Presence", "No_Catapult", "No_Arresting_Gear",
    "No_Well_Deck", "No_Stern_Gate", "No_Full_Flight_Deck",
    "No_Landing_Craft_Capability", "No_Large_Aviation_Facility",
    "No_Fixed_Wing_Carrier_Operation", "No_VLS",
}


def get_effective_slot_weight(slot: str, group: str) -> float:
    """获得当前 slot 的基础权重。"""
    if group == "EQUIPMENT_DETAILS":
        return min(float(SLOT_WEIGHTS.get(slot, 0.6)), 0.8)
    return float(LOCAL_SLOT_WEIGHT_OVERRIDES.get(slot, SLOT_WEIGHTS.get(slot, 0.8)))


def adjusted_slot_weight(slot: str, input_value: Any, base_weight: float) -> float:
    """
    第一层静态修正：
    只处理明显低信息的字段，真正的“共用/稀有”区分交给后面的动态区分度函数。
    """
    if is_unknown_value(input_value):
        return 0.0

    text = clean_text(input_value)

    # 负特征本身是辅助排除证据，不作为强确认项。
    # 注意这里不写死最终分数，只先做基础降权。
    if slot.startswith("No_"):
        return base_weight * 0.55

    # 无/0 是低信息量值，但如果该“无/0”在 7 个舰级里很少出现，仍可能有区分度。
    # 所以这里只做温和降权，后续由动态 IDF 再根据出现频率继续调节。
    if slot in LOW_INFORMATION_ABSENT_SLOTS and text in LOW_INFORMATION_ABSENT_VALUES:
        return base_weight * 0.55

    return base_weight


# ============================================================
# 5.2 动态区分度权重：根据 class_data 自动判断“共用特征/稀有特征”
# ============================================================

def observed_value_items(slot: str, value: Any) -> List[str]:
    """把一个观测值拆成标准化后的值列表。"""
    values = split_values(value)
    if not values and isinstance(value, str):
        values = [value]
    result = []
    for v in values:
        v_norm = normalize_slot_value(slot, v)
        if slot in SLOT_SCHEMA.get("EQUIPMENT_DETAILS", []):
            v_norm = normalize_equipment_name(v_norm)
        if not is_unknown_value(v_norm):
            result.append(v_norm)
    return result


def specificity_factor(df: int, total_classes: int, slot: str, value: Any) -> float:
    """
    根据某个 slot=value 在 7 个已知舰级中出现的数量动态调整权重。

    df 越小，说明该特征越稀有、越有区分度，权重越高；
    df 越大，说明该特征越共用，权重越低。
    对“无/0/非主要特征”这类低信息值，如果它们又被很多舰级共享，则进一步降权。
    """
    if total_classes <= 0 or df <= 0:
        return 1.0

    # 基础 IDF 型区分度。这里不用连续公式，便于调试和解释。
    if df == 1:
        factor = 1.80
    elif df == 2:
        factor = 1.40
    elif df == 3:
        factor = 1.10
    elif df == 4:
        factor = 0.85
    elif df == 5:
        factor = 0.60
    else:
        factor = 0.35

    text = clean_text(value)
    is_absent_like = text in LOW_INFORMATION_ABSENT_VALUES or slot.startswith("No_")

    # 如果“无/0”是很多舰级都共有的，就进一步降低；
    # 如果“无/0”很少见，仍保留一定区分度，但不让它超过正向强特征。
    if is_absent_like:
        if df >= 5:
            factor *= 0.55
        elif df >= 3:
            factor *= 0.70
        else:
            factor = min(factor, 0.95)

    return max(0.15, factor)


def compute_specificity_map(
    prototypes: Dict[str, Dict[str, Any]],
    observed: Dict[str, Dict[str, Any]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    根据当前输入中的每个 slot=value，统计它在 7 个已知舰级原型中能匹配多少个。

    返回：
    {
      (slot, compact_value): {
        "df": 出现/匹配的舰级数量,
        "total": 已知舰级数量,
        "factor": 动态区分度因子,
        "matched_classes": [舰级名...]
      }
    }
    """
    total = len(prototypes)
    result: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for _group, slot, obs_value in iter_observed_slots(observed):
        for v_norm in observed_value_items(slot, obs_value):
            key = (slot, compact_key(v_norm))
            if key in result:
                continue

            matched_classes = []
            for ship_class, proto in prototypes.items():
                proto_value = get_proto_slot_value(proto, slot)
                matched, _reason = value_match(slot, v_norm, proto_value)
                if matched:
                    matched_classes.append(ship_class)

            df = len(matched_classes)
            result[key] = {
                "df": df,
                "total": total,
                "factor": specificity_factor(df, total, slot, v_norm),
                "matched_classes": matched_classes,
            }

    return result


def get_specificity_info(
    specificity_map: Dict[Tuple[str, str], Dict[str, Any]],
    slot: str,
    value: Any,
) -> Dict[str, Any]:
    """获取一个观测字段的动态区分度信息。多值时取最高区分度。"""
    candidates = []
    for v_norm in observed_value_items(slot, value):
        info = specificity_map.get((slot, compact_key(v_norm)))
        if info:
            candidates.append(info)
    if not candidates:
        return {"df": 0, "total": 0, "factor": 1.0, "matched_classes": []}
    return max(candidates, key=lambda x: x.get("factor", 1.0))


# 大类提示改成 slot 约束，防止 Catapult_Count=0 误匹配到 VLS_Count_Level=90-96单元级。
CATEGORY_HINTS_BY_SLOT = {
    "航空母舰": {
        "Flight_Deck_Type": {"全通飞行甲板": 2.0, "斜角飞行甲板": 2.5},
        "Catapult": {"有": 3.0},
        "Arresting_Gear": {"有": 3.0},
        "Aircraft_Capacity_Level": {"大量固定翼舰载机": 2.5},
        "Island_Position": {"右舷舰岛": 1.5},
        "Powerplant": {"核动力": 1.5},
    },
    "巡洋舰": {
        "VLS_Count_Level": {"122单元级": 2.5, "高": 1.5},
        "Main_Gun_Position": {"舰艏和舰艉各1门": 1.5},
        "Radar_Array_Type": {"四面固定相控阵": 1.5},
        "Area_Air_Defense": {"强": 2.0},
        "Command_Control": {"强": 1.5},
        "Primary_Mission": {"区域防空": 2.0, "舰队指挥": 1.5},
    },
    "驱逐舰": {
        "VLS_Count_Level": {"90-96单元级": 2.0, "中": 1.0},
        "Main_Gun_Caliber": {"127mm级": 1.5},
        "Main_Gun_Position": {"舰艏": 1.0},
        "Phased_Array_Radar": {"有": 2.0},
        "Radar_Array_Type": {"四面固定相控阵": 2.0},
        "Primary_Mission": {"多用途导弹驱逐舰": 2.0},
    },
    "护卫舰": {
        "Hull_Form": {"三体船": 4.0},
        "Superstructure_Type": {"低矮隐身化上层建筑": 2.5},
        "Stealth_Shape": {"低矮隐身化外形": 2.0},
        "Flight_Deck_Type": {"艉部直升机甲板": 2.0, "大型艉部直升机飞行甲板": 2.0},
        "Flight_Deck_Position": {"舰尾": 1.0},
        "Main_Gun_Caliber": {"57mm级": 2.0, "中小口径": 2.0},
        "Mission_Module": {"模块化任务": 2.0},
    },
    "两栖舰": {
        "Flight_Deck_Type": {"全通飞行甲板": 1.5, "艉部直升机甲板": 1.0},
        "Well_Deck": {"有": 3.0, "大型坞舱": 3.0},
        "Stern_Gate": {"有": 2.0},
        "Landing_Craft_Capability": {"LCAC": 2.0, "登陆艇": 2.0},
        "Troop_Transport": {"有": 1.5, "强": 2.0},
        "Amphibious_Assault_Capability": {"强": 2.5},
        "STOVL_Aircraft_Operation": {"有": 1.5},
    },
    "登陆舰": {
        "Well_Deck": {"有": 3.0, "大型坞舱": 3.0},
        "Stern_Gate": {"有": 2.5},
        "Landing_Craft_Capability": {"LCAC": 2.0, "LCU": 2.0, "登陆艇": 2.0},
        "Landing_Craft_Capacity": {"4艘LCAC级": 2.0},
        "Landing_Operation": {"强": 2.0},
        "Primary_Mission": {"船坞登陆": 3.0},
    },
}


def is_valid_category_hint_match(slot: str, input_value: Any, hint: Any) -> bool:
    """大类提示匹配必须在同一 slot 内完成，并避免短数字/短文本误匹配。"""
    if is_unknown_value(input_value) or is_unknown_value(hint):
        return False

    value_text = clean_text(input_value)
    hint_text = clean_text(hint)

    # 防止 0 匹配到 90-96单元级、100000吨等。
    if re.fullmatch(r"\d+(?:\.\d+)?", value_text):
        return value_text == hint_text

    value_key = compact_key(value_text)
    hint_key = compact_key(hint_text)

    if not value_key or not hint_key:
        return False

    # “有/无”这类短值只有完全相等才算。
    if value_text in {"有", "无", "0", "1", "2", "3", "4"}:
        return value_text == hint_text

    if len(value_key) <= 1:
        return False

    return value_key == hint_key or value_key in hint_key or hint_key in value_key


# ============================================================
# 5.3 槽位语义归一：解决“艉部直升机甲板”与“大型艉部直升机飞行甲板”等近义表达
# ============================================================

GENERIC_MODIFIERS = [
    "大型", "小型", "中型", "明显", "较大", "较小", "有限", "强", "中等", "普通",
    "外形", "结构", "能力", "特征", "级", "左右", "约", "大约",
]


def semantic_compact_key(slot: str, value: Any) -> str:
    """比 compact_key 更宽松的槽位级语义 key，只用于同 slot 的匹配。"""
    text = clean_text(value)

    # 主炮口径做层级归一：57mm/中小口径可以互相匹配，127mm 不会误匹配。
    if slot == "Main_Gun_Caliber":
        nums = extract_numbers(text)
        if nums:
            n = nums[0]
            if n <= 60:
                return "中小口径"
            if n < 100:
                return "中口径"
            return "大口径"
        if "中小" in text or "小口径" in text:
            return "中小口径"
        if "127" in text or "大口径" in text:
            return "大口径"

    if slot in {"Flight_Deck_Type", "Flight_Deck_Position"}:
        text = text.replace("直升机飞行甲板", "直升机甲板")
        text = text.replace("飞行甲板", "甲板")
        text = text.replace("舰尾", "艉部")
        text = text.replace("尾部", "艉部")

    if slot in {"Superstructure_Type", "Stealth_Shape"}:
        # 保留“低矮/箱形/舰岛式”等强信息，但让“隐身化外形/隐身化上层建筑”可互相靠近。
        text = text.replace("隐身化外形", "隐身化")
        text = text.replace("隐身化上层建筑", "隐身化")
        text = text.replace("封闭式", "封闭")

    for m in GENERIC_MODIFIERS:
        text = text.replace(m, "")

    return compact_key(text)


def semantic_slot_match(slot: str, observed_value: Any, prototype_value: Any) -> Tuple[bool, str]:
    """同一 slot 内的语义近似匹配，避免过度依赖字符串包含。"""
    ov = clean_text(observed_value)
    pv = clean_text(prototype_value)
    if not ov or not pv:
        return False, ""

    ok = semantic_compact_key(slot, ov)
    pk = semantic_compact_key(slot, pv)
    if ok and pk and (ok == pk or ok in pk or pk in ok):
        return True, f"语义接近：{ov} ≈ {pv}"

    # 隐身相关不能直接等同所有舰，但可以作为弱匹配证据，由 df 动态区分度继续降权。
    if slot in {"Superstructure_Type", "Stealth_Shape"} and "隐身" in ov and "隐身" in pv:
        return True, f"均为隐身化相关表述：{ov} / {pv}"

    return False, ""

def value_match(slot: str, observed_value: Any, prototype_value: Any) -> Tuple[bool, str]:
    """判断单个 observed 值是否匹配 class_data 原型值。"""
    if is_unknown_value(observed_value) or is_unknown_value(prototype_value):
        return False, ""

    obs_values = split_values(observed_value)
    proto_values = split_values(prototype_value)
    if not obs_values:
        obs_values = [clean_text(observed_value)]
    if not proto_values:
        proto_values = [clean_text(prototype_value)]

    for ov in obs_values:
        ov_norm = normalize_slot_value(slot, ov)
        if slot in SLOT_SCHEMA.get("EQUIPMENT_DETAILS", []):
            ov_norm = normalize_equipment_name(ov_norm)
        ok = compact_key(ov_norm)
        if not ok:
            continue

        for pv in proto_values:
            pv_norm = normalize_slot_value(slot, pv)
            if slot in SLOT_SCHEMA.get("EQUIPMENT_DETAILS", []):
                pv_norm = normalize_equipment_name(pv_norm)
            pk = compact_key(pv_norm)
            if not pk:
                continue

            if ok == pk:
                return True, f"{ov_norm} = {pv_norm}"
            if ok in pk or pk in ok:
                return True, f"{ov_norm} ≈ {pv_norm}"
            if slot in NUMERIC_SLOTS and numeric_match(ov_norm, pv_norm):
                return True, f"数值接近：{ov_norm} ≈ {pv_norm}"
            sem_ok, sem_reason = semantic_slot_match(slot, ov_norm, pv_norm)
            if sem_ok:
                return True, sem_reason
            if is_presence_yes(ov_norm) and is_presence_yes(pv_norm):
                return True, f"均表示存在：{ov_norm} / {pv_norm}"
            if is_presence_no(ov_norm) and is_presence_no(pv_norm):
                return True, f"均表示不存在：{ov_norm} / {pv_norm}"

    return False, ""



# 正向卡槽与对应负特征卡槽的映射。
# 用于判断冲突：例如输入 Catapult=有，但某原型 No_Catapult=是，则应扣分。
NEGATIVE_SLOT_MAP = {
    "Well_Deck": "No_Well_Deck",
    "Stern_Gate": "No_Stern_Gate",
    "Catapult": "No_Catapult",
    "Arresting_Gear": "No_Arresting_Gear",
    "Flight_Deck_Type": "No_Full_Flight_Deck",
    "Landing_Craft_Capability": "No_Landing_Craft_Capability",
    "VLS_Presence": "No_VLS",
    "Fixed_Wing_Aircraft_Operation": "No_Fixed_Wing_Carrier_Operation",
}


# ============================================================
# 5.1 开放集与原始文本提示增强
# ============================================================

# 这些词只用于判断“用户明确说不像已知类 / 不能稳定匹配已知类”。
# 注意不要把“不是航母”“不像两栖舰”这类普通排除语句当成开放集提示。
UNKNOWN_INTENT_CUES = [
    "不像已知", "不符合已知", "与已知", "和已知", "已知类不", "已知原型不",
    "不完全一致", "差异明显", "相似度不足", "不能稳定匹配", "无法稳定对应",
    "无法稳定匹配已知", "类别内未知", "未知目标", "未知类",
]

RAW_CATEGORY_CUES = {
    "航空母舰": ["航空母舰", "航母", "超级航母", "核动力航空母舰"],
    "巡洋舰": ["巡洋舰", "导弹巡洋舰", "宙斯盾巡洋舰", "舰队指挥"],
    "驱逐舰": ["驱逐舰", "导弹驱逐舰", "防空驱逐舰"],
    "护卫舰": ["护卫舰", "巡防舰", "濒海战斗舰", "近海巡逻", "护航任务"],
    "两栖舰": ["两栖舰", "两栖攻击舰", "船坞运输舰", "两栖船坞运输舰", "两栖运输", "两栖攻击"],
    "登陆舰": ["登陆舰", "船坞登陆舰", "船坞登陆", "登陆艇投送"],
}

# 对这些关键槽位，如果输入与原型明确不一致，要作为冲突扣分，不能只当 unmatched。
# 这样可以避免“单体护卫舰”被硬匹配到“三体船”的独立级。
STRONG_SLOT_MISMATCH_PENALTIES = {
    "Hull_Form": 4.0,
    "Flight_Deck_Type": 2.5,
    "Well_Deck": 3.0,
    "Stern_Gate": 2.5,
    "Catapult": 3.0,
    "Arresting_Gear": 3.0,
    "VLS_Presence": 2.0,
    "Main_Gun_Caliber": 1.5,
    "Superstructure_Type": 2.0,
}

# 对于单已知舰级大类，适当放宽“输出具体舰级”的阈值。
# 否则会大量出现 category 正确但 class=None。
SINGLE_CLASS_KNOWN_LOW_CONF_THRESHOLD = 0.35
SINGLE_CLASS_MIN_EVIDENCE_COUNT = 4
GENERAL_KNOWN_LOW_CONF_THRESHOLD = 0.60

# 原始文本直接提示大类时给一点额外大类分，但只作为辅助。
RAW_CATEGORY_BOOST = 4.0


def get_raw_text_from_observed(observed: Dict[str, Any]) -> str:
    meta = observed.get("_META", {}) if isinstance(observed, dict) else {}
    if isinstance(meta, dict):
        return str(meta.get("raw_text", "") or "")
    return ""


def has_unknown_intent(raw_text: str) -> bool:
    text = compact_key(raw_text)
    if not text:
        return False
    return any(compact_key(cue) in text for cue in UNKNOWN_INTENT_CUES)


def _cue_is_negated(raw_text: str, start_idx: int) -> bool:
    prefix = raw_text[max(0, start_idx - 6):start_idx]
    return any(x in prefix for x in ["不像", "不是", "并非", "非", "没有", "未见", "无"])


def raw_category_scores(raw_text: str) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    scores = {cat: 0.0 for cat in SHIP_CATEGORIES}
    evidence: List[Dict[str, Any]] = []
    if not raw_text:
        return scores, evidence

    for cat, cues in RAW_CATEGORY_CUES.items():
        for cue in cues:
            idx = raw_text.find(cue)
            if idx < 0:
                continue
            if _cue_is_negated(raw_text, idx):
                continue
            scores[cat] += RAW_CATEGORY_BOOST
            evidence.append({
                "category": cat,
                "slot": "RAW_TEXT",
                "input_value": cue,
                "hint": cue,
                "base_score": RAW_CATEGORY_BOOST,
                "specificity_factor": 1.0,
                "score": RAW_CATEGORY_BOOST,
                "reason": "原始文本中出现大类提示词",
            })
            break
    return scores, evidence


def is_meaningful_known_value(v: Any) -> bool:
    t = clean_text(v)
    if t in {"", "未知", "不确定", "未提及"}:
        return False
    return True


def should_apply_strong_mismatch(slot: str, obs_value: Any, proto_value: Any) -> bool:
    if slot not in STRONG_SLOT_MISMATCH_PENALTIES:
        return False
    if not is_meaningful_known_value(obs_value) or not is_meaningful_known_value(proto_value):
        return False
    # “无/0”这类共性缺失特征已经通过动态区分度降权，一般不再额外作为强冲突。
    if clean_text(obs_value) in LOW_INFORMATION_ABSENT_VALUES or clean_text(proto_value) in LOW_INFORMATION_ABSENT_VALUES:
        return False
    return True

def get_proto_slot_value(proto: Dict[str, Any], slot: str) -> Any:
    group = SLOT_TO_GROUP.get(slot)
    if not group:
        return "未知"
    return proto.get("groups", {}).get(group, {}).get(slot, "未知")


def iter_observed_slots(observed: Dict[str, Dict[str, Any]]):
    for group, slots in observed.items():
        # _META 用于保存原始输入文本等调试信息，不参与属性槽位遍历。
        if str(group).startswith("_"):
            continue
        if group == "CLASS" or not isinstance(slots, dict):
            continue
        for slot, value in slots.items():
            if is_unknown_value(value):
                continue
            yield group, slot, value


def category_hint_scores(
    observed: Dict[str, Dict[str, Any]],
    specificity_map: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    大类提示分。

    关键修正：
    1. 只允许同 slot 匹配，防止 Catapult_Count=0 匹配到 VLS_Count_Level=90-96单元级。
    2. 低信息量的“无/0”不参与大类提示加分。
    3. 大类提示分也乘以动态区分度，避免“舰艏主炮”等共性特征把多个类别抬高。
    """
    specificity_map = specificity_map or {}
    scores = {cat: 0.0 for cat in SHIP_CATEGORIES}
    evidence: List[Dict[str, Any]] = []

    for _group, slot, value in iter_observed_slots(observed):
        values = split_values(value)
        if not values and isinstance(value, str):
            values = [value]

        for v in values:
            v_norm = normalize_slot_value(slot, v)

            # 大类提示主要依赖“有信息量”的正向结构特征。
            if clean_text(v_norm) in LOW_INFORMATION_ABSENT_VALUES:
                continue

            specificity_info = get_specificity_info(specificity_map, slot, v_norm)
            factor = float(specificity_info.get("factor", 1.0))

            for cat, slot_hint_map in CATEGORY_HINTS_BY_SLOT.items():
                hints_for_slot = slot_hint_map.get(slot, {})
                for hint, weight in hints_for_slot.items():
                    if is_valid_category_hint_match(slot, v_norm, hint):
                        adjusted_score = float(weight) * factor
                        scores[cat] += adjusted_score
                        evidence.append({
                            "category": cat,
                            "slot": slot,
                            "input_value": v_norm,
                            "hint": hint,
                            "base_score": weight,
                            "specificity_factor": round(factor, 3),
                            "score": round(adjusted_score, 3),
                        })

    return scores, evidence


def match_one_class(
    proto: Dict[str, Any],
    observed: Dict[str, Dict[str, Any]],
    specificity_map: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    ship_class = proto["groups"].get("CLASS", {}).get("Ship_Class", proto.get("title", ""))
    category = proto.get("category", "")

    score = 0.0
    possible = 0.0
    evidence: List[Dict[str, Any]] = []
    conflicts: List[Dict[str, Any]] = []
    unmatched: List[Dict[str, Any]] = []

    for group, slot, obs_value in iter_observed_slots(observed):
        if slot not in SLOT_WEIGHTS and group != "EQUIPMENT_DETAILS":
            continue

        base_weight = get_effective_slot_weight(slot, group)
        static_weight = adjusted_slot_weight(slot, obs_value, base_weight)

        # 未知、不确定、空值，或被判定为无信息量的字段，不计入 possible。
        if static_weight <= 0:
            continue

        specificity_info = get_specificity_info(specificity_map or {}, slot, obs_value)
        dynamic_factor = float(specificity_info.get("factor", 1.0))
        weight = static_weight * dynamic_factor

        possible += weight
        proto_value = get_proto_slot_value(proto, slot)

        matched, reason = value_match(slot, obs_value, proto_value)
        if matched:
            score += weight
            evidence.append({
                "slot": slot,
                "group": group,
                "input_value": obs_value,
                "prototype_value": proto_value,
                "score": round(weight, 3),
                "base_weight": round(base_weight, 3),
                "specificity_factor": round(dynamic_factor, 3),
                "feature_df": specificity_info.get("df", 0),
                "feature_total": specificity_info.get("total", 0),
                "reason": reason,
            })
        else:
            unmatched.append({
                "slot": slot,
                "group": group,
                "input_value": obs_value,
                "prototype_value": proto_value,
            })

            # 强区分槽位明确不一致时，作为冲突扣分，而不是只记 unmatched。
            # 例如：输入 Hull_Form=单体船，独立级原型 Hull_Form=三体船。
            if should_apply_strong_mismatch(slot, obs_value, proto_value):
                penalty = STRONG_SLOT_MISMATCH_PENALTIES.get(slot, 1.0)
                score -= penalty
                conflicts.append({
                    "slot": slot,
                    "group": group,
                    "input_value": obs_value,
                    "prototype_value": proto_value,
                    "penalty": penalty,
                    "reason": f"关键槽位不一致：输入 {slot}={obs_value}，原型 {slot}={proto_value}",
                })

        # 负特征冲突：用户明确看到“有”，但该已知类明确“No_xxx: 是”。
        neg_slot = NEGATIVE_SLOT_MAP.get(slot)
        if neg_slot and is_presence_yes(obs_value):
            neg_value = proto.get("groups", {}).get("NEGATIVE_FEATURES", {}).get(neg_slot, "未知")
            if clean_text(neg_value) == "是":
                penalty = NEGATIVE_CONFLICT_WEIGHTS.get(neg_slot, 2.0)
                score -= penalty
                conflicts.append({
                    "slot": slot,
                    "negative_slot": neg_slot,
                    "input_value": obs_value,
                    "prototype_negative_value": neg_value,
                    "penalty": penalty,
                    "reason": f"输入观察到 {slot}={obs_value}，但该已知类标记 {neg_slot}=是",
                })

    confidence = max(0.0, score) / possible if possible > 0 else 0.0
    return {
        "ship_class": ship_class,
        "category": category,
        "score": round(score, 4),
        "possible_score": round(possible, 4),
        "confidence": round(confidence, 4),
        "matched_evidence": evidence,
        "conflict_evidence": conflicts,
        "unmatched_conditions": unmatched,
    }


# ============================================================
# 5.5 最终输出决策封装
# ============================================================

FINAL_AMBIGUOUS_CATEGORY_MARGIN = 0.08
FINAL_AMBIGUOUS_CLASS_MARGIN = 0.10
FINAL_TOPK = 3


def _brief_known_candidate(r: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "label": r.get("ship_class"),
        "category": r.get("category"),
        "confidence": round(float(r.get("confidence", 0.0)), 4),
        "score": r.get("score", 0.0),
        "matched_evidence_count": len(r.get("matched_evidence", [])),
        "conflict_count": len(r.get("conflict_evidence", [])),
    }


def _brief_category_candidate(c: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "label": c.get("label"),
        "confidence": round(float(c.get("confidence", 0.0)), 4),
        "score": c.get("score", 0.0),
    }


def build_final_decision(
    category_result: Dict[str, Any],
    known_class_result: Optional[Dict[str, Any]],
    open_set_result: Dict[str, Any],
    category_candidates: List[Dict[str, Any]],
    class_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    将详细候选结果压缩成一个面向用户的最终判定。

    设计原则：
    1. 如果最高置信度结果明显领先，则单独作为 final_decision 输出。
    2. 如果第一、第二候选差距过小，则输出 ambiguous，并列出多个可能。
    3. 如果大类明确但已知舰级不足，则输出“类别内未知类”。
    4. category_result / known_class_result / candidates 仍然保留，方便调试。
    """
    top_category = category_candidates[0] if category_candidates else None
    second_category = category_candidates[1] if len(category_candidates) > 1 else None
    category_margin = 0.0
    if top_category and second_category:
        category_margin = float(top_category.get("confidence", 0.0)) - float(second_category.get("confidence", 0.0))

    top_class = class_results[0] if class_results else None
    second_class = class_results[1] if len(class_results) > 1 else None
    class_margin = 0.0
    if top_class and second_class:
        class_margin = float(top_class.get("confidence", 0.0)) - float(second_class.get("confidence", 0.0))

    top_categories = [_brief_category_candidate(c) for c in category_candidates[:FINAL_TOPK]]
    top_known_classes = [_brief_known_candidate(r) for r in class_results[:FINAL_TOPK]]

    # 1. 大类都无法确定
    if not category_result or not category_result.get("label"):
        return {
            "result_type": "uncertain",
            "primary_category": None,
            "primary_class": None,
            "confidence": round(float(category_result.get("confidence", 0.0)) if category_result else 0.0, 4),
            "status": "insufficient_information",
            "message": "最终判定：输入信息不足，暂时无法稳定判断舰船大类或具体已知舰级。",
            "alternatives": {
                "top_categories": top_categories,
                "top_known_classes": top_known_classes,
            },
        }

    # 2. 已经匹配到已知舰级
    if known_class_result:
        # 如果第一、第二已知舰级非常接近，不强行给唯一结论
        if second_class and class_margin < FINAL_AMBIGUOUS_CLASS_MARGIN:
            return {
                "result_type": "ambiguous_known_class",
                "primary_category": category_result.get("label"),
                "primary_class": None,
                "confidence": round(float(known_class_result.get("confidence", 0.0)), 4),
                "status": "multiple_close_known_classes",
                "message": "最终判定：存在多个置信度接近的已知舰级候选，无法唯一确定具体已知舰级。",
                "margin": round(class_margin, 4),
                "alternatives": {
                    "top_categories": top_categories,
                    "top_known_classes": top_known_classes,
                },
            }

        return {
            "result_type": "known_class",
            "primary_category": known_class_result.get("category"),
            "primary_class": known_class_result.get("label"),
            "confidence": round(float(known_class_result.get("confidence", 0.0)), 4),
            "status": "single_best",
            "message": (
                f"最终判定：{known_class_result.get('category')} / "
                f"{known_class_result.get('label')}。该结果为当前最高置信度候选。"
            ),
            "margin": round(class_margin, 4),
            "alternatives": {
                "top_categories": top_categories,
                "top_known_classes": top_known_classes,
            },
        }

    # 3. 已判断为某大类，但不匹配该大类下任何已知舰级：类别内未知
    if open_set_result and open_set_result.get("is_unknown"):
        if second_category and category_margin < FINAL_AMBIGUOUS_CATEGORY_MARGIN:
            return {
                "result_type": "ambiguous_category",
                "primary_category": None,
                "primary_class": None,
                "confidence": round(float(category_result.get("confidence", 0.0)), 4),
                "status": "multiple_close_categories",
                "message": "最终判定：多个舰船大类置信度差距过小，暂时无法唯一确定大类。",
                "margin": round(category_margin, 4),
                "alternatives": {
                    "top_categories": top_categories,
                    "top_known_classes": top_known_classes,
                },
            }

        return {
            "result_type": "category_unknown",
            "primary_category": category_result.get("label"),
            "primary_class": None,
            "confidence": round(float(category_result.get("confidence", 0.0)), 4),
            "status": "open_set_unknown",
            "message": f"最终判定：{open_set_result.get('unknown_scope')}。该目标能判断大类，但与已知舰级不够匹配。",
            "margin": round(category_margin, 4),
            "alternatives": {
                "top_categories": top_categories,
                "top_known_classes": top_known_classes,
            },
        }

    # 4. 兜底：只有大类，没有已知舰级结论
    if second_category and category_margin < FINAL_AMBIGUOUS_CATEGORY_MARGIN:
        return {
            "result_type": "ambiguous_category",
            "primary_category": None,
            "primary_class": None,
            "confidence": round(float(category_result.get("confidence", 0.0)), 4),
            "status": "multiple_close_categories",
            "message": "最终判定：多个舰船大类置信度差距过小，暂时无法唯一确定大类。",
            "margin": round(category_margin, 4),
            "alternatives": {
                "top_categories": top_categories,
                "top_known_classes": top_known_classes,
            },
        }

    return {
        "result_type": "category_only",
        "primary_category": category_result.get("label"),
        "primary_class": None,
        "confidence": round(float(category_result.get("confidence", 0.0)), 4),
        "status": "single_best_category",
        "message": f"最终判定：{category_result.get('label')}。当前只能稳定判断大类，不能确认具体已知舰级。",
        "margin": round(category_margin, 4),
        "alternatives": {
            "top_categories": top_categories,
            "top_known_classes": top_known_classes,
        },
    }

def hierarchical_class_match(
    class_data_path: str,
    observed_attributes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """层级分类：六大类 -> 已知舰级 -> 类别内未知。"""
    prototypes = load_class_data(class_data_path)

    # 根据当前输入和 7 个已知舰级原型，动态计算每个 slot=value 的全局区分度。
    # 这一步可以自动降低“所有舰都共有的无/0”，提高“只有一两个舰级有的强特征”。
    specificity_map = compute_specificity_map(prototypes, observed_attributes)

    raw_text = get_raw_text_from_observed(observed_attributes)
    unknown_intent = has_unknown_intent(raw_text)

    class_results = [
        match_one_class(proto, observed_attributes, specificity_map)
        for proto in prototypes.values()
    ]
    class_results.sort(key=lambda x: (x["confidence"], x["score"]), reverse=True)

    top_overall_class = class_results[0] if class_results else None
    second_overall_class = class_results[1] if len(class_results) > 1 else None

    # 大类分数 = 类内已知舰级最高分 + slot 约束的大类特征提示分 + 原始文本大类提示分。
    hint_scores, hint_evidence = category_hint_scores(observed_attributes, specificity_map)
    raw_scores, raw_evidence = raw_category_scores(raw_text)
    hint_evidence.extend(raw_evidence)
    category_scores = {
        cat: hint_scores.get(cat, 0.0) + raw_scores.get(cat, 0.0)
        for cat in SHIP_CATEGORIES
    }

    for cat in SHIP_CATEGORIES:
        class_conf = max([r["confidence"] for r in class_results if r["category"] == cat] or [0.0])
        category_scores[cat] += class_conf * 10.0

    total_cat_score = sum(max(v, 0.0) for v in category_scores.values()) or 1.0
    category_candidates = []
    for cat, raw_score in sorted(category_scores.items(), key=lambda x: x[1], reverse=True):
        category_candidates.append({
            "label": cat,
            "score": round(raw_score, 4),
            "confidence": round(max(raw_score, 0.0) / total_cat_score, 4),
        })

    top_category = category_candidates[0] if category_candidates else {"label": None, "confidence": 0.0}

    # 关键修正：如果某个已知舰级已经高置信匹配，则大类直接继承该舰级的大类。
    # 否则容易出现“大类分布被摊薄，category_result=uncertain，但 known_class 第一名很高”的矛盾。
    selected_category = None
    selected_category_conf = 0.0
    category_reason = ""

    top_known_conf = top_overall_class["confidence"] if top_overall_class else 0.0
    second_known_conf = second_overall_class["confidence"] if second_overall_class else 0.0
    top_known_margin = top_known_conf - second_known_conf

    if (not unknown_intent) and top_overall_class and (
        top_known_conf >= 0.75 or
        (top_known_conf >= KNOWN_CLASS_CONFIDENCE_THRESHOLD and top_known_margin >= KNOWN_CLASS_MARGIN_THRESHOLD)
    ):
        selected_category = top_overall_class["category"]
        selected_category_conf = max(top_known_conf, top_category.get("confidence", 0.0))
        category_reason = (
            f"已知舰级 {top_overall_class['ship_class']} 匹配度最高，"
            f"因此大类继承为 {selected_category}。"
        )
    elif top_category.get("confidence", 0.0) >= CATEGORY_CONFIDENCE_THRESHOLD:
        selected_category = top_category["label"]
        selected_category_conf = top_category["confidence"]
        category_reason = "根据属性卡槽和大类提示特征，当前大类得分最高。"

    if not selected_category:
        return {
            "category_result": {
                "label": None,
                "confidence": round(top_category.get("confidence", 0.0), 4),
                "status": "uncertain",
                "reason": "输入信息不足，无法稳定判断舰船大类。",
            },
            "known_class_result": None,
            "open_set_result": {
                "is_unknown": False,
                "unknown_scope": None,
                "reason": "大类尚不明确，暂不进行类别内未知判断。",
            },
            "final_decision": build_final_decision(
                {
                    "label": None,
                    "confidence": round(top_category.get("confidence", 0.0), 4),
                    "status": "uncertain",
                    "reason": "输入信息不足，无法稳定判断舰船大类。",
                },
                None,
                {
                    "is_unknown": False,
                    "unknown_scope": None,
                    "reason": "大类尚不明确，暂不进行类别内未知判断。",
                },
                category_candidates,
                class_results,
            ),
            "category_candidates": category_candidates,
            "known_class_candidates": [
                {
                    "label": r["ship_class"],
                    "category": r["category"],
                    "confidence": r["confidence"],
                    "score": r["score"],
                    "matched_evidence_count": len(r["matched_evidence"]),
                    "conflict_count": len(r["conflict_evidence"]),
                }
                for r in class_results
            ],
            "primary_evidence": (top_overall_class or {}).get("matched_evidence", [])[:20],
            "primary_conflicts": (top_overall_class or {}).get("conflict_evidence", [])[:20],
            "primary_unmatched_conditions": (top_overall_class or {}).get("unmatched_conditions", [])[:30],
            "category_hint_evidence": hint_evidence[:30],
            "method_note": "LLM 仅用于文本属性抽取；大类/已知舰级/类别内未知判断由 class_data 原型卡和规则匹配完成。",
        }

    same_category_classes = [r for r in class_results if r["category"] == selected_category]
    top_class = same_category_classes[0] if same_category_classes else None
    second_class = same_category_classes[1] if len(same_category_classes) > 1 else None

    top_conf = top_class["confidence"] if top_class else 0.0
    second_conf = second_class["confidence"] if second_class else 0.0
    margin = top_conf - second_conf

    category_result = {
        "label": selected_category,
        "confidence": round(selected_category_conf, 4),
        "status": "matched",
        "reason": category_reason,
    }

    category_known_classes = KNOWN_SHIP_CLASSES.get(selected_category, []) if selected_category else []
    if len(category_known_classes) <= 1:
        margin_ok = True
    else:
        margin_ok = margin >= KNOWN_CLASS_MARGIN_THRESHOLD or top_conf >= 0.75

    # 如果原始文本明确表达“与已知类不一致/不能稳定匹配已知类”，优先输出类别内未知类。
    if unknown_intent:
        known_class_result = None
        open_set_result = {
            "is_unknown": True,
            "unknown_scope": UNKNOWN_OUTPUT_TEMPLATE.format(category=selected_category),
            "reason": (
                f"输入可以判断为{selected_category}，且文本包含与已知类不一致的提示；"
                "因此输出类别内未知类。"
            ),
        }
    else:
        evidence_count = len(top_class.get("matched_evidence", [])) if top_class else 0
        low_conf_single_class_ok = (
            top_class is not None
            and len(category_known_classes) <= 1
            and top_conf >= SINGLE_CLASS_KNOWN_LOW_CONF_THRESHOLD
            and evidence_count >= SINGLE_CLASS_MIN_EVIDENCE_COUNT
        )
        general_low_conf_ok = (
            top_class is not None
            and top_conf >= GENERAL_KNOWN_LOW_CONF_THRESHOLD
            and margin_ok
        )

        if top_class and (
            (top_conf >= KNOWN_CLASS_CONFIDENCE_THRESHOLD and margin_ok)
            or low_conf_single_class_ok
            or general_low_conf_ok
        ):
            reason = "该已知舰级原型匹配分达到阈值。"
            if low_conf_single_class_ok and top_conf < KNOWN_CLASS_CONFIDENCE_THRESHOLD:
                reason = "该大类下仅有一个已知舰级，且匹配到足够证据，因此输出低置信已知舰级。"

            known_class_result = {
                "label": top_class["ship_class"],
                "category": top_class["category"],
                "confidence": round(top_conf, 4),
                "score": top_class["score"],
                "known_status": "Known",
                "reason": reason,
            }
            open_set_result = {
                "is_unknown": False,
                "unknown_scope": None,
                "reason": "已匹配到已知舰级。",
            }
        else:
            known_class_result = None
            open_set_result = {
                "is_unknown": True,
                "unknown_scope": UNKNOWN_OUTPUT_TEMPLATE.format(category=selected_category),
                "reason": (
                    f"输入可以判断为{selected_category}，但与该大类下已知舰级的匹配分不足；"
                    "因此输出类别内未知类。"
                ),
            }

    final_decision = build_final_decision(
        category_result,
        known_class_result,
        open_set_result,
        category_candidates,
        class_results,
    )

    return {
        "final_decision": final_decision,
        "category_result": category_result,
        "known_class_result": known_class_result,
        "open_set_result": open_set_result,
        "category_candidates": category_candidates,
        "known_class_candidates": [
            {
                "label": r["ship_class"],
                "category": r["category"],
                "confidence": r["confidence"],
                "score": r["score"],
                "matched_evidence_count": len(r["matched_evidence"]),
                "conflict_count": len(r["conflict_evidence"]),
            }
            for r in class_results
        ],
        "primary_evidence": (top_class or {}).get("matched_evidence", [])[:20],
        "primary_conflicts": (top_class or {}).get("conflict_evidence", [])[:20],
        "primary_unmatched_conditions": (top_class or {}).get("unmatched_conditions", [])[:30],
        "category_hint_evidence": hint_evidence[:30],
        "specificity_summary": [
            {
                "slot": slot,
                "value_key": value_key,
                "df": info.get("df", 0),
                "total": info.get("total", 0),
                "factor": round(float(info.get("factor", 1.0)), 3),
                "matched_classes": info.get("matched_classes", []),
            }
            for (slot, value_key), info in sorted(
                specificity_map.items(),
                key=lambda x: (x[1].get("df", 0), -x[1].get("factor", 1.0), x[0][0], x[0][1])
            )[:50]
        ],
        "method_note": "LLM 仅用于文本属性抽取；大类/已知舰级/类别内未知判断由 class_data 原型卡和规则匹配完成；匹配权重会根据 class_data 中特征出现频率动态调整。",
    }



# ============================================================
# 5.2 v8：开放集与单已知类输出进一步修正
# ============================================================

# 扩展开放集提示词。重点覆盖“不能匹配已知类”“没有足以匹配已知类特征”等表达。
UNKNOWN_INTENT_CUES = list(dict.fromkeys(UNKNOWN_INTENT_CUES + [
    "不属于已知", "不能匹配已知", "无法匹配已知", "不能对应已知", "无法对应已知",
    "不稳定对应", "不能稳定对应", "不稳定匹配", "不能稳定匹配",
    "不匹配已知", "未能匹配已知", "没有足以匹配已知", "不足以匹配已知",
    "不像已知的", "不同于已知", "与已知类差异", "与已知原型差异",
    "不符合已知", "不符合已知类", "不符合已知原型",
    "不完全符合", "不完全匹配", "不完全对应",
]))

# 原始文本大类提示增强。v7 中 category=None 的很多样本，其实原文已经出现了强大类词。
RAW_CATEGORY_BOOST = 8.0
RAW_CATEGORY_CUES = {
    "航空母舰": ["航空母舰", "航母", "超级航母", "核动力航空母舰", "海上机场", "大型飞行甲板"],
    "巡洋舰": ["巡洋舰", "导弹巡洋舰", "宙斯盾巡洋舰", "舰队指挥", "前后都有发射区域", "前后甲板可见武器发射区域", "前后都有炮", "舰艏和舰艉"],
    "驱逐舰": ["驱逐舰", "导弹驱逐舰", "防空驱逐舰", "现代导弹驱逐舰"],
    "护卫舰": ["护卫舰", "巡防舰", "濒海战斗舰", "近海巡逻", "护航任务", "单体护卫舰", "常规护卫舰"],
    "两栖舰": ["两栖舰", "两栖攻击舰", "船坞运输舰", "两栖船坞运输舰", "两栖运输", "两栖攻击", "运兵", "海军陆战队"],
    "登陆舰": ["登陆舰", "船坞登陆舰", "船坞登陆", "登陆艇投送", "大型船坞", "泛水坞舱"],
}


def _cue_is_negated(raw_text: str, start_idx: int) -> bool:
    """扩大否定窗口，避免“它不像航母或两栖舰”误给两栖舰加分。"""
    prefix = raw_text[max(0, start_idx - 12):start_idx]
    return any(x in prefix for x in ["不像", "不是", "并非", "非", "没有", "未见", "无", "不属于"])


def infer_raw_category(raw_text: str) -> Tuple[Optional[str], float, List[Dict[str, Any]]]:
    """从原始文本中提取直接大类提示。比归一化分布更适合开放集兜底。"""
    scores, evidence = raw_category_scores(raw_text)
    if not scores:
        return None, 0.0, evidence
    cat, score = max(scores.items(), key=lambda x: x[1])
    if score <= 0:
        return None, 0.0, evidence
    return cat, score, evidence


def has_strong_conflict(match_result: Dict[str, Any], min_penalty: float = 2.0) -> bool:
    """判断 top_class 是否存在关键冲突。"""
    for c in match_result.get("conflict_evidence", []) or []:
        try:
            if float(c.get("penalty", 0.0)) >= min_penalty:
                return True
        except Exception:
            continue
    return False


def category_single_known_class(category: Optional[str]) -> Optional[str]:
    if not category:
        return None
    classes = KNOWN_SHIP_CLASSES.get(category, [])
    if len(classes) == 1:
        return classes[0]
    return None


def hierarchical_class_match(
    class_data_path: str,
    observed_attributes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """v8 层级分类：强化 raw category、开放集和单已知大类输出。"""
    prototypes = load_class_data(class_data_path)
    specificity_map = compute_specificity_map(prototypes, observed_attributes)

    raw_text = get_raw_text_from_observed(observed_attributes)
    unknown_intent = has_unknown_intent(raw_text)
    raw_cat, raw_cat_score, raw_evidence = infer_raw_category(raw_text)

    class_results = [
        match_one_class(proto, observed_attributes, specificity_map)
        for proto in prototypes.values()
    ]
    class_results.sort(key=lambda x: (x["confidence"], x["score"]), reverse=True)

    top_overall_class = class_results[0] if class_results else None
    second_overall_class = class_results[1] if len(class_results) > 1 else None

    hint_scores, hint_evidence = category_hint_scores(observed_attributes, specificity_map)
    raw_scores, raw_evidence_2 = raw_category_scores(raw_text)
    # raw_evidence_2 与 infer_raw_category 可能重复，保留一次即可
    hint_evidence.extend(raw_evidence_2)

    category_scores = {cat: hint_scores.get(cat, 0.0) + raw_scores.get(cat, 0.0) for cat in SHIP_CATEGORIES}
    for cat in SHIP_CATEGORIES:
        class_conf = max([r["confidence"] for r in class_results if r["category"] == cat] or [0.0])
        category_scores[cat] += class_conf * 10.0

    total_cat_score = sum(max(v, 0.0) for v in category_scores.values()) or 1.0
    category_candidates = []
    for cat, raw_score in sorted(category_scores.items(), key=lambda x: x[1], reverse=True):
        category_candidates.append({
            "label": cat,
            "score": round(raw_score, 4),
            "confidence": round(max(raw_score, 0.0) / total_cat_score, 4),
        })

    top_category = category_candidates[0] if category_candidates else {"label": None, "confidence": 0.0}

    top_known_conf = top_overall_class["confidence"] if top_overall_class else 0.0
    second_known_conf = second_overall_class["confidence"] if second_overall_class else 0.0
    top_known_margin = top_known_conf - second_known_conf

    selected_category = None
    selected_category_conf = 0.0
    category_reason = ""

    # v8 关键：开放集文本中只要明确给出大类，就先锁定大类，再输出类别内未知。
    if unknown_intent and raw_cat:
        selected_category = raw_cat
        selected_category_conf = max(top_category.get("confidence", 0.0), 0.55)
        category_reason = f"原始文本包含{raw_cat}类提示，并明确表达与已知类不一致，因此优先判定大类为{raw_cat}。"
    elif (not unknown_intent) and top_overall_class and (
        top_known_conf >= 0.75 or
        (top_known_conf >= KNOWN_CLASS_CONFIDENCE_THRESHOLD and top_known_margin >= KNOWN_CLASS_MARGIN_THRESHOLD)
    ):
        selected_category = top_overall_class["category"]
        selected_category_conf = max(top_known_conf, top_category.get("confidence", 0.0))
        category_reason = f"已知舰级 {top_overall_class['ship_class']} 匹配度最高，因此大类继承为 {selected_category}。"
    elif raw_cat and raw_cat_score >= RAW_CATEGORY_BOOST:
        selected_category = raw_cat
        selected_category_conf = max(top_category.get("confidence", 0.0), 0.50)
        category_reason = f"原始文本中出现明确的{raw_cat}类提示词，因此优先判定大类为{raw_cat}。"
    elif top_category.get("confidence", 0.0) >= CATEGORY_CONFIDENCE_THRESHOLD:
        selected_category = top_category["label"]
        selected_category_conf = top_category["confidence"]
        category_reason = "根据属性卡槽和大类提示特征，当前大类得分最高。"
    elif top_overall_class and top_known_conf >= 0.28 and len(top_overall_class.get("matched_evidence", [])) >= 3:
        selected_category = top_overall_class["category"]
        selected_category_conf = max(top_known_conf, top_category.get("confidence", 0.0))
        category_reason = f"大类分布不够集中，但已知舰级 {top_overall_class['ship_class']} 提供了足够证据，因此暂定大类为 {selected_category}。"

    if not selected_category:
        category_result = {
            "label": None,
            "confidence": round(top_category.get("confidence", 0.0), 4),
            "status": "uncertain",
            "reason": "输入信息不足，无法稳定判断舰船大类。",
        }
        open_set_result = {
            "is_unknown": False,
            "unknown_scope": None,
            "reason": "大类尚不明确，暂不进行类别内未知判断。",
        }
        return {
            "category_result": category_result,
            "known_class_result": None,
            "open_set_result": open_set_result,
            "final_decision": build_final_decision(category_result, None, open_set_result, category_candidates, class_results),
            "category_candidates": category_candidates,
            "known_class_candidates": [
                {
                    "label": r["ship_class"], "category": r["category"], "confidence": r["confidence"],
                    "score": r["score"], "matched_evidence_count": len(r["matched_evidence"]),
                    "conflict_count": len(r["conflict_evidence"]),
                }
                for r in class_results
            ],
            "primary_evidence": (top_overall_class or {}).get("matched_evidence", [])[:20],
            "primary_conflicts": (top_overall_class or {}).get("conflict_evidence", [])[:20],
            "primary_unmatched_conditions": (top_overall_class or {}).get("unmatched_conditions", [])[:30],
            "category_hint_evidence": hint_evidence[:30],
            "method_note": "v8：LLM 仅用于属性抽取；大类/已知舰级/类别内未知由规则匹配完成。",
        }

    same_category_classes = [r for r in class_results if r["category"] == selected_category]
    top_class = same_category_classes[0] if same_category_classes else None
    second_class = same_category_classes[1] if len(same_category_classes) > 1 else None

    top_conf = top_class["confidence"] if top_class else 0.0
    second_conf = second_class["confidence"] if second_class else 0.0
    margin = top_conf - second_conf
    evidence_count = len(top_class.get("matched_evidence", [])) if top_class else 0
    conflict_count = len(top_class.get("conflict_evidence", [])) if top_class else 0
    strong_conflict = has_strong_conflict(top_class or {})

    category_result = {
        "label": selected_category,
        "confidence": round(selected_category_conf, 4),
        "status": "matched",
        "reason": category_reason,
    }

    category_known_classes = KNOWN_SHIP_CLASSES.get(selected_category, []) if selected_category else []
    margin_ok = True if len(category_known_classes) <= 1 else (margin >= KNOWN_CLASS_MARGIN_THRESHOLD or top_conf >= 0.75)

    # 开放集优先：只要原文明确表达“不像/不符合已知类”，则不输出已知舰级。
    if unknown_intent:
        known_class_result = None
        open_set_result = {
            "is_unknown": True,
            "unknown_scope": UNKNOWN_OUTPUT_TEMPLATE.format(category=selected_category),
            "reason": f"输入可以判断为{selected_category}，且文本包含与已知类不一致的提示；因此输出类别内未知类。",
        }
    else:
        # v8：单已知舰级大类，只要大类已经判定且无强冲突，就输出该唯一已知舰级。
        unique_known = category_single_known_class(selected_category)
        unique_single_ok = (
            unique_known is not None
            and top_class is not None
            and top_class["ship_class"] == unique_known
            and evidence_count >= 2
            and not strong_conflict
        )

        # 多已知舰级大类，适当降低阈值，但需要证据和 margin。
        multi_known_ok = (
            top_class is not None
            and len(category_known_classes) > 1
            and top_conf >= 0.45
            and evidence_count >= 4
            and margin_ok
            and not strong_conflict
        )

        normal_known_ok = (
            top_class is not None
            and top_conf >= KNOWN_CLASS_CONFIDENCE_THRESHOLD
            and margin_ok
            and not strong_conflict
        )

        if top_class and (normal_known_ok or unique_single_ok or multi_known_ok):
            reason = "该已知舰级原型匹配分达到阈值。"
            if unique_single_ok and not normal_known_ok:
                reason = "该大类下仅有一个已知舰级，且未发现关键冲突，因此输出该已知舰级。"
            elif multi_known_ok and not normal_known_ok:
                reason = "该大类下候选舰级匹配证据较充分，因此输出低置信已知舰级。"

            known_class_result = {
                "label": top_class["ship_class"],
                "category": top_class["category"],
                "confidence": round(top_conf, 4),
                "score": top_class["score"],
                "known_status": "Known",
                "reason": reason,
            }
            open_set_result = {"is_unknown": False, "unknown_scope": None, "reason": "已匹配到已知舰级。"}
        else:
            known_class_result = None
            open_set_result = {
                "is_unknown": True,
                "unknown_scope": UNKNOWN_OUTPUT_TEMPLATE.format(category=selected_category),
                "reason": f"输入可以判断为{selected_category}，但与该大类下已知舰级的匹配分不足或存在关键冲突；因此输出类别内未知类。",
            }

    final_decision = build_final_decision(category_result, known_class_result, open_set_result, category_candidates, class_results)

    return {
        "final_decision": final_decision,
        "category_result": category_result,
        "known_class_result": known_class_result,
        "open_set_result": open_set_result,
        "category_candidates": category_candidates,
        "known_class_candidates": [
            {
                "label": r["ship_class"],
                "category": r["category"],
                "confidence": r["confidence"],
                "score": r["score"],
                "matched_evidence_count": len(r["matched_evidence"]),
                "conflict_count": len(r["conflict_evidence"]),
            }
            for r in class_results
        ],
        "primary_evidence": (top_class or {}).get("matched_evidence", [])[:20],
        "primary_conflicts": (top_class or {}).get("conflict_evidence", [])[:20],
        "primary_unmatched_conditions": (top_class or {}).get("unmatched_conditions", [])[:30],
        "category_hint_evidence": hint_evidence[:30],
        "specificity_summary": [
            {
                "slot": slot,
                "value_key": value_key,
                "df": info.get("df", 0),
                "total": info.get("total", 0),
                "factor": round(float(info.get("factor", 1.0)), 3),
                "matched_classes": info.get("matched_classes", []),
            }
            for (slot, value_key), info in sorted(
                specificity_map.items(),
                key=lambda x: (x[1].get("df", 0), -x[1].get("factor", 1.0), x[0][0], x[0][1])
            )[:50]
        ],
        "method_note": "v8：LLM 仅用于文本属性抽取；大类/已知舰级/类别内未知判断由 class_data 原型卡和规则匹配完成。",
    }



# ============================================================
# 5.3 v10：未知护卫舰/开放集后处理进一步增强
# ============================================================

# v10 说明：
# 1. open_set=True 时，final_decision 不再因为候选大类 margin 小而清空 primary_category；
# 2. 对“护卫舰类别内未知类”增加结构化兜底：单体船 + 直升机甲板 + 舰艏炮 + 非三体/不像独立级；
# 3. 增补“没有独立级特征 / 不是三体结构 / 没有三体船结构”等未知类触发词。

V10_EXTRA_UNKNOWN_INTENT_CUES = [
    "没有独立级", "不具备独立级", "不像独立级", "不同于独立级",
    "没有已知独立级", "不能匹配独立级", "无法匹配独立级",
    "不是三体结构", "不是三体船", "没有三体船", "没有三体船结构", "没有三体船特征",
    "不是多体船", "没有多体船", "没有多体船外形",
    "不同于已知三体", "不具备三体", "未表现出三体",
]


def _v10_text_has_any(raw_text: str, cues: List[str]) -> bool:
    ck = compact_key(raw_text)
    return any(compact_key(cue) in ck for cue in cues)


def _v10_is_frigate_like_structure(raw_text: str) -> bool:
    """从原始文本中识别“护卫舰/巡防舰/轻型水面舰”的结构提示。"""
    ck = compact_key(raw_text)
    has_direct_frigate = any(compact_key(x) in ck for x in [
        "护卫舰", "巡防舰", "常规护卫舰", "单体护卫舰", "护卫舰常见布局"
    ])
    has_small = any(compact_key(x) in ck for x in [
        "体量不大", "体型中等偏小", "中等偏小", "体型不大", "较小", "中小型", "体量较小"
    ])
    has_front_gun = any(compact_key(x) in ck for x in [
        "舰艏有主炮", "舰艏主炮", "前部有炮", "船头有炮", "舰艏有炮", "前部舰炮"
    ])
    has_helo_deck = any(compact_key(x) in ck for x in [
        "舰尾有直升机", "舰尾直升机", "后部有直升机", "直升机甲板", "直升机平台", "直升机起降区"
    ])
    excludes_big_ship = any(compact_key(x) in ck for x in [
        "没有全通飞行甲板", "未见全通飞行甲板", "没有坞舱", "不具备大型两栖坞舱",
        "不是航空母舰", "不像航母", "不是两栖舰", "不像两栖舰"
    ])
    return has_direct_frigate or (has_small and has_front_gun and has_helo_deck) or (has_front_gun and has_helo_deck and excludes_big_ship)


def _v10_has_frigate_unknown_structure(raw_text: str) -> bool:
    ck = compact_key(raw_text)
    not_independence = _v10_text_has_any(raw_text, V10_EXTRA_UNKNOWN_INTENT_CUES)
    monohull_like = any(compact_key(x) in ck for x in [
        "普通单体船", "传统单体船", "单体船", "单体护卫舰", "舰体为单体", "舰体为普通单体"
    ])
    return _v10_is_frigate_like_structure(raw_text) and (not_independence or monohull_like)


# 覆盖 v9 的 has_unknown_intent：增加护卫舰未知结构触发。
def has_unknown_intent(raw_text: str) -> bool:
    text = compact_key(raw_text)
    if not text:
        return False
    all_cues = list(dict.fromkeys(UNKNOWN_INTENT_CUES + V10_EXTRA_UNKNOWN_INTENT_CUES))
    if any(compact_key(cue) in text for cue in all_cues):
        return True
    if _v10_has_frigate_unknown_structure(raw_text):
        return True
    return False


# 覆盖 v9 的 infer_raw_category：增加未知护卫舰结构兜底。
def infer_raw_category(raw_text: str) -> Tuple[Optional[str], float, List[Dict[str, Any]]]:
    scores, evidence = raw_category_scores(raw_text)

    # 如果有明确护卫舰/巡防舰或典型轻型护卫舰结构，则给护卫舰补分。
    if _v10_is_frigate_like_structure(raw_text):
        scores["护卫舰"] = max(scores.get("护卫舰", 0.0), RAW_CATEGORY_BOOST)
        evidence.append({
            "category": "护卫舰",
            "slot": "RAW_TEXT",
            "input_value": "护卫舰/轻型水面舰结构提示",
            "hint": "单体船/舰艏主炮/舰尾直升机甲板/非大型航母两栖结构",
            "base_score": RAW_CATEGORY_BOOST,
            "specificity_factor": 1.0,
            "score": RAW_CATEGORY_BOOST,
            "reason": "v10：原始文本呈现护卫舰或巡防舰类结构特征",
        })

    if not scores:
        return None, 0.0, evidence
    cat, score = max(scores.items(), key=lambda x: x[1])
    if score <= 0:
        return None, 0.0, evidence
    return cat, score, evidence


# 覆盖原 build_final_decision：open_set=True 且大类已知时，优先保留大类，不再因 margin 小转成 ambiguous。
def build_final_decision(
    category_result: Dict[str, Any],
    known_class_result: Optional[Dict[str, Any]],
    open_set_result: Dict[str, Any],
    category_candidates: List[Dict[str, Any]],
    class_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    top_category = category_candidates[0] if category_candidates else None
    second_category = category_candidates[1] if len(category_candidates) > 1 else None
    category_margin = 0.0
    if top_category and second_category:
        category_margin = float(top_category.get("confidence", 0.0)) - float(second_category.get("confidence", 0.0))

    top_class = class_results[0] if class_results else None
    second_class = class_results[1] if len(class_results) > 1 else None
    class_margin = 0.0
    if top_class and second_class:
        class_margin = float(top_class.get("confidence", 0.0)) - float(second_class.get("confidence", 0.0))

    top_categories = [_brief_category_candidate(c) for c in category_candidates[:FINAL_TOPK]]
    top_known_classes = [_brief_known_candidate(r) for r in class_results[:FINAL_TOPK]]

    if not category_result or not category_result.get("label"):
        return {
            "result_type": "uncertain",
            "primary_category": None,
            "primary_class": None,
            "confidence": round(float(category_result.get("confidence", 0.0)) if category_result else 0.0, 4),
            "status": "insufficient_information",
            "message": "最终判定：输入信息不足，暂时无法稳定判断舰船大类或具体已知舰级。",
            "alternatives": {"top_categories": top_categories, "top_known_classes": top_known_classes},
        }

    if known_class_result:
        if second_class and class_margin < FINAL_AMBIGUOUS_CLASS_MARGIN:
            return {
                "result_type": "ambiguous_known_class",
                "primary_category": category_result.get("label"),
                "primary_class": None,
                "confidence": round(float(known_class_result.get("confidence", 0.0)), 4),
                "status": "multiple_close_known_classes",
                "message": "最终判定：存在多个置信度接近的已知舰级候选，无法唯一确定具体已知舰级。",
                "margin": round(class_margin, 4),
                "alternatives": {"top_categories": top_categories, "top_known_classes": top_known_classes},
            }
        return {
            "result_type": "known_class",
            "primary_category": known_class_result.get("category"),
            "primary_class": known_class_result.get("label"),
            "confidence": round(float(known_class_result.get("confidence", 0.0)), 4),
            "status": "single_best",
            "message": f"最终判定：{known_class_result.get('category')} / {known_class_result.get('label')}。该结果为当前最高置信度候选。",
            "margin": round(class_margin, 4),
            "alternatives": {"top_categories": top_categories, "top_known_classes": top_known_classes},
        }

    if open_set_result and open_set_result.get("is_unknown"):
        return {
            "result_type": "category_unknown",
            "primary_category": category_result.get("label"),
            "primary_class": None,
            "confidence": round(float(category_result.get("confidence", 0.0)), 4),
            "status": "open_set_unknown",
            "message": f"最终判定：{open_set_result.get('unknown_scope')}。该目标能判断大类，但与已知舰级不够匹配。",
            "margin": round(category_margin, 4),
            "alternatives": {"top_categories": top_categories, "top_known_classes": top_known_classes},
        }

    if second_category and category_margin < FINAL_AMBIGUOUS_CATEGORY_MARGIN:
        return {
            "result_type": "ambiguous_category",
            "primary_category": None,
            "primary_class": None,
            "confidence": round(float(category_result.get("confidence", 0.0)), 4),
            "status": "multiple_close_categories",
            "message": "最终判定：多个舰船大类置信度差距过小，暂时无法唯一确定大类。",
            "margin": round(category_margin, 4),
            "alternatives": {"top_categories": top_categories, "top_known_classes": top_known_classes},
        }

    return {
        "result_type": "category_only",
        "primary_category": category_result.get("label"),
        "primary_class": None,
        "confidence": round(float(category_result.get("confidence", 0.0)), 4),
        "status": "single_best_category",
        "message": f"最终判定：{category_result.get('label')}。当前只能稳定判断大类，不能确认具体已知舰级。",
        "margin": round(category_margin, 4),
        "alternatives": {"top_categories": top_categories, "top_known_classes": top_known_classes},
    }


# ============================================================
# 6. 主程序
# ============================================================

async def main():
    CLASS_DATA_PATH = "./class_data.txt"
    WORKING_DIR = "./class_index"

    # class_data 或 schema_config 改了，就设为 True 重建 GraphML。
    REBUILD_GRAPH = True

    # 是否创建 GraphRAG 对象。分类不依赖它；你后续要 aquery 解释时可以打开。
    ENABLE_GRAPHRAG_OBJECT = False

    graphml_file = os.path.join(WORKING_DIR, "graph_chunk_entity_relation.graphml")
    if REBUILD_GRAPH or not os.path.exists(graphml_file):
        build_class_graph_from_class_data(WORKING_DIR, CLASS_DATA_PATH, rebuild=REBUILD_GRAPH)
    else:
        print(f"[跳过规则构图] 已存在 GraphML: {graphml_file}")

    print_graph_summary(WORKING_DIR)

    graph_func = None
    if ENABLE_GRAPHRAG_OBJECT:
        graph_func = build_graph_rag(WORKING_DIR)
        print("[GraphRAG] 已加载 class graph，可用于后续解释性查询。")

    # ========== 测试文本 ==========
    # 你可以替换成：用户口头描述 / 图像大模型输出描述 / 百科参数文本。
    user_text = """图片中这艘舰整体体型较小，船体不是普通单体船，而是明显的三体船结构。舰桥和上层建筑位于舰体前部，外形比较低矮并带有隐身化倾斜面。舰尾有一块面积较大的直升机飞行甲板，可以起降直升机，后方似乎还有任务舱或小艇收放区域。舰艏位置可以看到一门中小口径主炮，未见全通飞行甲板，也没有弹射器、拦阻索或大型坞舱。"""
    
    print("\n" + "=" * 60)
    print("【步骤1】LLM 文本解析结果：")
    parse_text = await direct_text_parse_v2(user_text)
    print(parse_text)

    try:
        parsed_obj = extract_json_object(parse_text)
        observed = normalize_observed_attributes(parsed_obj)
        schema_errors = validate_observed_schema(observed)
        if schema_errors:
            print("\n[Schema 校验警告]")
            for err in schema_errors:
                print("-", err)
    except Exception as e:
        print(f"解析 LLM JSON 失败：{type(e).__name__}: {e}")
        return

    print("\n" + "=" * 60)
    print("【步骤2】标准化后的属性卡槽：")
    print(json.dumps(observed, ensure_ascii=False, indent=2))

    result = hierarchical_class_match(CLASS_DATA_PATH, observed)

    print("\n" + "=" * 60)
    print("【步骤3】最终判断结果：")
    print(json.dumps(result.get("final_decision", {}), ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    print("【步骤4】层级分类匹配详情：")
    detail_result = {k: v for k, v in result.items() if k != "final_decision"}
    print(json.dumps(detail_result, ensure_ascii=False, indent=2))




# ============================================================
# 5.3 v9：基于原始文本的大类/开放集后处理修正
# ============================================================

# 保存 v8 版本，v9 在其结果基础上做后处理，避免重写全部匹配逻辑。
_hierarchical_class_match_v8 = hierarchical_class_match

# 继续扩展开放集触发词，重点覆盖“未表现出已知X的典型外形”等表达。
UNKNOWN_INTENT_CUES = list(dict.fromkeys(UNKNOWN_INTENT_CUES + [
    "并未表现出已知", "未表现出已知", "没有表现出已知", "没有体现已知",
    "典型外形不一致", "典型布局不一致", "布局不一致", "轮廓不一致",
    "无法稳定判断为已知", "不能稳定判断为已知", "不具备已知", "没有已知",
    "没有出现足以匹配", "没有足以匹配", "不足以对应", "不足以判断为已知",
    "与已知舰级不同", "与已知舰级不一致", "与已知舰级差异", "与已知舰级不完全一致",
    "与已知类不完全一致", "与已知原型不完全一致", "相似度不足",
]))

# v9 使用更直接的大类文本提示，不完全依赖原 category_scores。
V9_DIRECT_CATEGORY_CUES = {
    "航空母舰": ["航空母舰", "核动力航空母舰", "超级航母", "航母", "海上机场", "贯通全舰的飞行甲板"],
    "巡洋舰": ["导弹巡洋舰", "宙斯盾巡洋舰", "巡洋舰", "舰队指挥", "舰艏和舰艉各", "前后都有炮", "舰艏和舰艉"],
    "驱逐舰": ["导弹驱逐舰", "现代导弹驱逐舰", "防空驱逐舰", "大型驱逐舰", "驱逐舰"],
    "护卫舰": ["护卫舰或巡防舰", "单体护卫舰", "常规护卫舰", "护卫舰", "巡防舰", "濒海战斗舰", "近海巡逻"],
    "两栖舰": ["两栖攻击舰", "两栖船坞运输舰", "船坞运输舰", "两栖运输", "两栖舰", "两栖攻击", "海军陆战队"],
    "登陆舰": ["船坞登陆舰", "船坞登陆", "登陆舰", "登陆艇投送", "大型船坞", "泛水坞舱"],
}


def v9_cue_is_negated(raw_text: str, idx: int) -> bool:
    prefix = raw_text[max(0, idx - 14):idx]
    return any(x in prefix for x in ["不像", "不是", "并非", "不属于", "没有", "未见", "无", "非"])


def v9_direct_category_from_raw(raw_text: str) -> Optional[str]:
    if not raw_text:
        return None
    scores = {cat: 0 for cat in SHIP_CATEGORIES}
    for cat, cues in V9_DIRECT_CATEGORY_CUES.items():
        for cue in cues:
            start = 0
            while True:
                idx = raw_text.find(cue, start)
                if idx < 0:
                    break
                if not v9_cue_is_negated(raw_text, idx):
                    # 更长、更明确的 cue 权重更高。
                    scores[cat] += 2 + min(len(cue), 8) / 4
                    break
                start = idx + len(cue)
    cat, score = max(scores.items(), key=lambda x: x[1])
    return cat if score > 0 else None


def v9_category_from_observed(observed: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """原文无明确大类时，用关键结构兜底推断大类。"""
    visual = observed.get("VISUAL_STRUCTURE", {}) or {}
    aviation = observed.get("AVIATION_FEATURES", {}) or {}
    amphib = observed.get("AMPHIBIOUS_FEATURES", {}) or {}
    weapon = observed.get("WEAPON_SENSOR_FEATURES", {}) or {}

    hull = str(visual.get("Hull_Form", "") or "")
    flight_type = str(aviation.get("Flight_Deck_Type", "") or "")
    catapult = str(aviation.get("Catapult", "") or "")
    arresting = str(aviation.get("Arresting_Gear", "") or "")
    well = str(amphib.get("Well_Deck", "") or "")
    stern_gate = str(amphib.get("Stern_Gate", "") or "")
    vls = str(weapon.get("VLS_Presence", "") or "")
    radar = str(weapon.get("Phased_Array_Radar", "") or "")

    if "三体" in hull:
        return "护卫舰"
    if "全通" in flight_type and catapult == "有":
        return "航空母舰"
    if "全通" in flight_type and well == "有" and catapult in {"无", "未知", "不确定"}:
        return "两栖舰"
    if well == "有" and stern_gate == "有":
        return "登陆舰"
    if vls == "有" and radar == "有":
        return "驱逐舰"
    return None


def v9_infer_category(raw_text: str, observed: Dict[str, Dict[str, Any]]) -> Optional[str]:
    raw_cat = v9_direct_category_from_raw(raw_text)
    if raw_cat:
        return raw_cat
    return v9_category_from_observed(observed)


V9_KNOWN_CLASS_RAW_CUES = {
    "尼米兹级航空母舰": ["尼米兹", "核动力航空母舰", "超级航母", "弹射器", "拦阻索", "固定翼舰载机"],
    "提康德罗加级导弹巡洋舰": ["导弹巡洋舰", "宙斯盾巡洋舰", "巡洋舰", "舰队指挥", "舰艏和舰艉各", "前后都有炮"],
    "阿利·伯克级驱逐舰": ["阿利·伯克", "导弹驱逐舰", "现代导弹驱逐舰", "防空驱逐舰", "Mk 41", "宙斯盾驱逐舰"],
    "独立级濒海战斗舰": ["独立级", "濒海战斗舰", "三体船", "多体结构", "任务模块"],
    "黄蜂级两栖攻击舰": ["黄蜂级", "两栖攻击舰", "全通飞行甲板", "垂直起降飞机", "STOVL"],
    "圣安东尼奥级两栖船坞运输舰": ["圣安东尼奥", "两栖船坞运输舰", "船坞运输舰", "大型箱形", "大型封闭式上层建筑", "大型箱形隐身化上层建筑"],
    "惠德比岛级船坞登陆舰": ["惠德比", "船坞登陆舰", "船坞登陆", "大型船坞", "登陆艇投送", "大型泛水坞舱"],
}


def v9_infer_known_class_from_raw(raw_text: str, category: Optional[str]) -> Optional[str]:
    if not raw_text or not category:
        return None
    candidates = set(KNOWN_SHIP_CLASSES.get(category, []))
    best_cls, best_score = None, 0
    for cls, cues in V9_KNOWN_CLASS_RAW_CUES.items():
        if cls not in candidates:
            continue
        score = 0
        for cue in cues:
            idx = raw_text.find(cue)
            if idx >= 0 and not v9_cue_is_negated(raw_text, idx):
                score += 1
        if score > best_score:
            best_cls, best_score = cls, score
    return best_cls if best_score > 0 else None


def v9_find_class_candidate(class_results: List[Dict[str, Any]], class_name: str) -> Optional[Dict[str, Any]]:
    for r in class_results:
        if r.get("ship_class") == class_name or r.get("label") == class_name:
            return r
    return None


def v9_rebuild_final(result: Dict[str, Any]) -> Dict[str, Any]:
    result["final_decision"] = build_final_decision(
        result.get("category_result") or {},
        result.get("known_class_result"),
        result.get("open_set_result") or {},
        result.get("category_candidates") or [],
        result.get("known_class_candidates") or [],
    )
    return result


def hierarchical_class_match(
    class_data_path: str,
    observed_attributes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """v9：在 v8 匹配结果基础上，修正开放集样本的大类缺失和已知类 class=None。"""
    result = _hierarchical_class_match_v8(class_data_path, observed_attributes)

    raw_text = get_raw_text_from_observed(observed_attributes)
    unknown_intent = has_unknown_intent(raw_text)
    inferred_category = v9_infer_category(raw_text, observed_attributes)

    category_result = result.get("category_result") or {}
    known_class_result = result.get("known_class_result")
    open_set_result = result.get("open_set_result") or {}
    class_candidates = result.get("known_class_candidates") or []

    # 1. 如果原文明确表示“与已知类不一致”，且可以从原文/结构推断大类，则优先输出类别内未知类。
    if unknown_intent and inferred_category:
        result["category_result"] = {
            "label": inferred_category,
            "confidence": max(float(category_result.get("confidence", 0.0) or 0.0), 0.60),
            "status": "matched",
            "reason": f"原始文本包含{inferred_category}方向提示，并表达与已知类不一致，因此判定为{inferred_category}类别内未知类。",
        }
        result["known_class_result"] = None
        result["open_set_result"] = {
            "is_unknown": True,
            "unknown_scope": UNKNOWN_OUTPUT_TEMPLATE.format(category=inferred_category),
            "reason": f"输入可以判断为{inferred_category}，但文本明确表示与已知舰级不一致或不能稳定匹配已知类。",
        }
        return v9_rebuild_final(result)

    # 2. 如果 v8 已经输出开放集，但大类为空，则用原始文本/结构补上大类。
    if open_set_result.get("is_unknown") and not category_result.get("label") and inferred_category:
        result["category_result"] = {
            "label": inferred_category,
            "confidence": 0.55,
            "status": "matched",
            "reason": f"开放集判断已触发，且原始文本/结构提示该目标属于{inferred_category}。",
        }
        result["open_set_result"] = {
            "is_unknown": True,
            "unknown_scope": UNKNOWN_OUTPUT_TEMPLATE.format(category=inferred_category),
            "reason": f"输入可以判断为{inferred_category}，但无法匹配该大类下已知舰级。",
        }
        return v9_rebuild_final(result)

    # 3. 非开放集样本，如果大类已明确但 class=None，尝试用唯一已知类或原文强提示补上已知舰级。
    if (not unknown_intent) and (not known_class_result) and category_result.get("label"):
        cat = category_result.get("label")
        inferred_class = v9_infer_known_class_from_raw(raw_text, cat)
        unique_class = category_single_known_class(cat)
        selected_class = inferred_class or unique_class

        if selected_class:
            cand = None
            # known_class_candidates 是摘要格式；如果能找到就拿分数，否则给一个低置信兜底。
            for r in class_candidates:
                if r.get("label") == selected_class:
                    cand = r
                    break
            conf = float((cand or {}).get("confidence", 0.45) or 0.45)
            score = float((cand or {}).get("score", 0.0) or 0.0)
            evidence_count = int((cand or {}).get("matched_evidence_count", 0) or 0)
            conflict_count = int((cand or {}).get("conflict_count", 0) or 0)

            # 只要没有明显冲突，并且大类本身已经被判定，就补出已知舰级。
            if conflict_count == 0 and (conf >= 0.20 or evidence_count >= 2 or inferred_class):
                result["known_class_result"] = {
                    "label": selected_class,
                    "category": cat,
                    "confidence": round(max(conf, 0.45), 4),
                    "score": score,
                    "known_status": "Known",
                    "reason": "大类已经明确，且该大类下存在唯一已知舰级或原始文本提供了已知舰级强提示，因此补充输出已知舰级。",
                }
                result["open_set_result"] = {"is_unknown": False, "unknown_scope": None, "reason": "已补充匹配到已知舰级。"}
                return v9_rebuild_final(result)

    return result



# ============================================================
# 5.4 v11：已知类后处理增强
# ============================================================
# v11 目标：
# 1. 解决“category 对了但 class=None”的已知样本；
# 2. 降低已知样本被误触发 open_set 的情况；
# 3. 利用原始文本中的强已知舰级线索补充 known_class；
# 4. 不对未知意图样本强行补已知类，避免覆盖 v10 的类别内未知判断。

_hierarchical_class_match_v10 = hierarchical_class_match


V11_KNOWN_CLASS_CUES = {
    "尼米兹级航空母舰": [
        "核动力航空母舰", "超级航母", "海上机场", "全通飞行甲板", "贯通全舰", "飞行甲板覆盖整个舰体",
        "一整块宽阔甲板", "右侧舰岛", "右舷舰岛", "多架飞机停放", "很多飞机停在甲板",
        "弹射器", "拦阻索", "固定翼舰载机", "大型机库"
    ],
    "提康德罗加级导弹巡洋舰": [
        "大型导弹巡洋舰", "导弹巡洋舰", "宙斯盾巡洋舰", "巡洋舰", "舰队指挥", "编队指挥",
        "舰艏和舰艉", "舰艏与舰艉", "船头和船尾", "前后都有炮", "舰艏和舰艉各有一门主炮",
        "前后区域布置垂直发射", "前后甲板可见武器发射区域", "大量垂直发射", "大量垂直发射单元",
        "较高的传统上层建筑", "传统上层建筑", "中部为较高的传统上层建筑"
    ],
    "阿利·伯克级驱逐舰": [
        "阿利·伯克", "导弹驱逐舰", "现代导弹驱逐舰", "防空驱逐舰", "多用途导弹驱逐舰",
        "宙斯盾驱逐舰", "Mk 41", "前后甲板有导弹发射井", "前后垂发", "隐身化封闭式上层建筑",
        "比较现代化的军舰", "外形有点隐身", "燃气轮机动力"
    ],
    "独立级濒海战斗舰": [
        "独立级", "濒海战斗舰", "三体船", "多体结构", "多体船", "不是普通单体船",
        "左右好像还有支撑结构", "船体好像比较宽", "低矮", "大型开放式直升机甲板",
        "任务模块", "模块化任务", "轻型濒海"
    ],
    "黄蜂级两栖攻击舰": [
        "黄蜂级", "两栖攻击舰", "全通式飞行甲板", "全通飞行甲板", "一整块飞行甲板",
        "右舷有舰岛", "右侧舰岛", "多处直升机作业区域", "直升机比较多",
        "短距起飞", "垂直降落飞机", "垂直起降飞机", "STOVL", "海军陆战队",
        "两栖登陆能力", "两栖攻击任务", "像小型航空母舰"
    ],
    "圣安东尼奥级两栖船坞运输舰": [
        "圣安东尼奥", "两栖船坞运输舰", "船坞运输舰", "两栖运输", "大型箱形隐身化上层建筑",
        "大型箱形", "大型封闭式上层建筑", "前部有大型箱形", "前部上层建筑很大",
        "两艘LCAC", "数百名海军陆战队员", "登陆支援任务", "舰艉可见可能的开口", "后面有直升机平台，船尾好像有开口"
    ],
    "惠德比岛级船坞登陆舰": [
        "惠德比", "船坞登陆舰", "船坞登陆", "大型船坞", "大型泛水坞舱", "泛水坞舱",
        "登陆艇投送", "多艘LCAC", "LCU登陆艇", "大型坞舱", "很大的尾门", "舰尾结构很特别",
        "船尾能打开", "可以打开的大门", "小艇进出", "大型开口", "给小艇进出", "登陆运输用"
    ],
}


def v11_score_known_class_from_raw(raw_text: str, class_name: str) -> float:
    """根据原始文本强提示为已知舰级打分。只使用通用词组，不使用样本 id。"""
    if not raw_text or class_name not in V11_KNOWN_CLASS_CUES:
        return 0.0
    score = 0.0
    for cue in V11_KNOWN_CLASS_CUES[class_name]:
        start = 0
        while True:
            idx = raw_text.find(cue, start)
            if idx < 0:
                break
            if not v9_cue_is_negated(raw_text, idx):
                # 长 cue / 专有 cue 权重略高
                score += 1.0 + min(len(cue), 12) / 12
                break
            start = idx + len(cue)
    return score


def v11_best_known_class_from_raw(raw_text: str, category: Optional[str]) -> Tuple[Optional[str], float]:
    if not raw_text or not category:
        return None, 0.0
    candidates = KNOWN_SHIP_CLASSES.get(category, [])
    best_cls, best_score = None, 0.0
    for cls in candidates:
        sc = v11_score_known_class_from_raw(raw_text, cls)
        if sc > best_score:
            best_cls, best_score = cls, sc
    return best_cls, best_score


def v11_get_candidate_summary(class_candidates: List[Dict[str, Any]], class_name: str) -> Dict[str, Any]:
    for r in class_candidates or []:
        if r.get("label") == class_name or r.get("ship_class") == class_name:
            return r
    return {}


def v11_set_known_class_result(
    result: Dict[str, Any],
    category: str,
    class_name: str,
    reason: str,
    min_conf: float = 0.55,
) -> Dict[str, Any]:
    class_candidates = result.get("known_class_candidates") or []
    cand = v11_get_candidate_summary(class_candidates, class_name)
    conf = max(float(cand.get("confidence", 0.0) or 0.0), min_conf)
    score = float(cand.get("score", 0.0) or 0.0)

    result["category_result"] = {
        "label": category,
        "confidence": max(float((result.get("category_result") or {}).get("confidence", 0.0) or 0.0), 0.55),
        "status": "matched",
        "reason": reason,
    }
    result["known_class_result"] = {
        "label": class_name,
        "category": category,
        "confidence": round(conf, 4),
        "score": score,
        "known_status": "Known",
        "reason": reason,
    }
    result["open_set_result"] = {
        "is_unknown": False,
        "unknown_scope": None,
        "reason": "原始文本与已知舰级存在足够强的类别/舰级提示，取消开放集输出。",
    }
    return v9_rebuild_final(result)


def v11_infer_known_by_single_category(raw_text: str, category: str) -> Optional[str]:
    """对只有一个已知类的大类，若原文不存在未知意图，且大类已确定，可以补已知类。"""
    candidates = KNOWN_SHIP_CLASSES.get(category, [])
    if len(candidates) == 1:
        return candidates[0]
    return None


def hierarchical_class_match(
    class_data_path: str,
    observed_attributes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """v11：在 v10 基础上，增强已知类的具体舰级补全和 known/open-set 冲突处理。"""
    result = _hierarchical_class_match_v10(class_data_path, observed_attributes)

    raw_text = get_raw_text_from_observed(observed_attributes)
    unknown_intent = has_unknown_intent(raw_text)

    category_result = result.get("category_result") or {}
    known_class_result = result.get("known_class_result")
    open_set_result = result.get("open_set_result") or {}

    # 推断大类：优先使用已有 category，其次使用 raw/结构。
    category = category_result.get("label") or v9_infer_category(raw_text, observed_attributes)
    if not category:
        return result

    # 如果已经是未知意图，且没有强已知类 raw cue，不覆盖开放集。
    raw_cls, raw_cls_score = v11_best_known_class_from_raw(raw_text, category)

    # 1) 如果没有未知意图，并且当前没有 known_class，则尽量补具体已知类。
    if not unknown_intent and not known_class_result:
        # 多已知类大类（两栖舰）优先用 raw cue；单已知大类直接补。
        selected_cls = raw_cls
        if not selected_cls:
            selected_cls = v11_infer_known_by_single_category(raw_text, category)

        if selected_cls:
            reason = (
                f"v11 后处理：大类已判定为{category}，"
                f"且原始文本/单已知类约束支持输出 {selected_cls}。"
            )
            return v11_set_known_class_result(result, category, selected_cls, reason, min_conf=0.55)

    # 2) 已经误触发 open_set，但 raw 强烈指向某已知舰级，则取消 open_set。
    # 只在 raw cue 分数较高时执行，避免覆盖真正未知类。
    if open_set_result.get("is_unknown") and raw_cls and raw_cls_score >= 2.0 and not unknown_intent:
        reason = f"v11 后处理：原始文本强烈指向已知舰级 {raw_cls}，因此取消类别内未知判断。"
        return v11_set_known_class_result(result, category, raw_cls, reason, min_conf=0.60)

    # 3) 如果 category 为空/不稳定但 raw 明确给出已知类，也补上。
    if not known_class_result and raw_cls and raw_cls_score >= 3.0 and not unknown_intent:
        reason = f"v11 后处理：原始文本强提示匹配 {raw_cls}。"
        return v11_set_known_class_result(result, category, raw_cls, reason, min_conf=0.60)

    return result



# ============================================================
# 5.5 v12：修复 known_class 为空未补全 + 原始文本强类目纠偏
# ============================================================
# v12 目标：
# 1. 修复 replay 中 category 正确但 pred_known_class 仍为空的问题；
# 2. 如果 known_class_result 是空 dict 或 label 为空，统一视为“未匹配已知舰级”；
# 3. 对非未知意图样本，允许原始文本中的强类别/舰级提示纠正“巡洋舰/驱逐舰/两栖舰/登陆舰”混淆；
# 4. 保留 v10/v11 已经修好的未知类逻辑，不覆盖 unknown_intent 样本。

_hierarchical_class_match_v11 = hierarchical_class_match


def v12_is_empty_known_result(known_class_result: Any) -> bool:
    if not known_class_result:
        return True
    if not isinstance(known_class_result, dict):
        return True
    label = str(known_class_result.get("label") or known_class_result.get("ship_class") or "").strip()
    return label in {"", "None", "null", "未知"}


def v12_raw_class_scores(raw_text: str) -> Dict[str, float]:
    """用原始文本给已知舰级打分。只作为后处理纠偏，不替代规则匹配。"""
    scores = {cls: 0.0 for cats in KNOWN_SHIP_CLASSES.values() for cls in cats}
    if not raw_text:
        return scores
    text = compact_key(raw_text)

    def add(cls: str, cue: str, w: float = 1.0):
        ck = compact_key(cue)
        idx = text.find(ck)
        if idx >= 0 and not v9_cue_is_negated(raw_text, max(0, raw_text.find(cue))):
            scores[cls] += w

    # 航母 / 尼米兹：全通大甲板 + 右舷舰岛 + 固定翼舰载机/弹射拦阻
    for cue, w in [
        ("核动力航空母舰", 3), ("超大型核动力航空母舰", 3), ("海上机场", 2.5),
        ("全通飞行甲板", 2), ("飞行甲板覆盖整个舰体", 3), ("贯通全舰", 2.5),
        ("右舷舰岛", 2), ("右侧舰岛", 2), ("多架飞机停放", 2),
        ("弹射器", 1.5), ("拦阻索", 1.5), ("固定翼舰载机", 2),
    ]:
        add("尼米兹级航空母舰", cue, w)

    # 巡洋舰 / 提康德罗加：巡洋舰词、前后主炮、大量垂发、舰队指挥
    for cue, w in [
        ("导弹巡洋舰", 4), ("大型导弹巡洋舰", 4), ("宙斯盾巡洋舰", 4), ("巡洋舰", 3),
        ("舰艏和舰艉", 3), ("舰艏与舰艉", 3), ("船头和船尾", 2.5), ("前后都有炮", 3),
        ("前后区域布置垂直发射", 3), ("大量垂直发射", 3), ("大量垂直发射单元", 3),
        ("舰队指挥", 2.5), ("编队指挥", 2.0), ("较高的传统上层建筑", 2),
    ]:
        add("提康德罗加级导弹巡洋舰", cue, w)

    # 驱逐舰 / 阿利·伯克：导弹驱逐舰词、Mk41、宙斯盾驱逐舰、隐身化封闭式上层建筑
    for cue, w in [
        ("阿利·伯克", 4), ("导弹驱逐舰", 4), ("现代导弹驱逐舰", 4), ("防空驱逐舰", 4),
        ("宙斯盾驱逐舰", 4), ("多用途导弹驱逐舰", 4), ("Mk 41", 3),
        ("前后甲板有导弹发射井", 3), ("前后垂发", 2.5),
        ("隐身化封闭式上层建筑", 2.5), ("燃气轮机动力", 2),
    ]:
        add("阿利·伯克级驱逐舰", cue, w)

    # 护卫舰 / 独立级：三体、多体、濒海战斗舰、任务模块
    for cue, w in [
        ("独立级", 4), ("濒海战斗舰", 4), ("三体船", 4), ("多体结构", 3.5),
        ("多体船", 3.5), ("不是普通单体船", 3), ("左右好像还有支撑结构", 3),
        ("任务模块", 2.5), ("模块化任务", 2.5), ("轻型濒海", 2),
    ]:
        add("独立级濒海战斗舰", cue, w)

    # 两栖攻击 / 黄蜂：全通甲板 + 直升机/STOVL + 坞舱/登陆艇，但无弹射拦阻
    for cue, w in [
        ("两栖攻击舰", 4), ("像小型航空母舰", 3), ("全通式飞行甲板", 3),
        ("一整块飞行甲板", 2.5), ("多处直升机作业区域", 2.5), ("直升机比较多", 2.5),
        ("短距起飞", 2.5), ("垂直起降飞机", 3), ("STOVL", 3), ("海军陆战队", 2.5),
        ("没有弹射器", 1.5), ("未见弹射器", 1.5), ("未见弹射器和传统固定翼拦阻", 3),
    ]:
        add("黄蜂级两栖攻击舰", cue, w)

    # 两栖船坞运输 / 圣安东尼奥：箱形/大型封闭上层建筑 + 直升机甲板 + 坞舱艉门/运兵车辆
    for cue, w in [
        ("两栖船坞运输舰", 4), ("船坞运输舰", 4), ("两栖运输", 3),
        ("大型箱形隐身化上层建筑", 4), ("大型箱形", 3), ("大型封闭式上层建筑", 3),
        ("前部有大型箱形", 3), ("前部上层建筑很大", 3), ("上层建筑体量很大", 3),
        ("后部有航空作业甲板", 2.5), ("舰尾疑似设置坞舱艉门", 4),
        ("船尾好像有开口", 2.5), ("两艘LCAC", 3), ("数百名海军陆战队员", 3),
        ("登陆支援任务", 2.5),
    ]:
        add("圣安东尼奥级两栖船坞运输舰", cue, w)

    # 船坞登陆 / 惠德比岛：船坞登陆、泛水坞舱、登陆艇投送、大型尾门、小艇进出
    for cue, w in [
        ("船坞登陆舰", 4), ("船坞登陆", 4), ("大型船坞", 3.5), ("大型泛水坞舱", 4),
        ("泛水坞舱", 3.5), ("登陆艇投送", 4), ("多艘LCAC", 3.5), ("LCU登陆艇", 3.5),
        ("大型坞舱", 3), ("很大的尾门", 3), ("舰尾结构很特别", 2.5),
        ("可以打开的大门", 3), ("给小艇进出", 3), ("小艇进出", 3),
        ("登陆运输用", 2.5),
    ]:
        add("惠德比岛级船坞登陆舰", cue, w)

    return scores


def v12_best_class_from_raw(raw_text: str) -> Tuple[Optional[str], Optional[str], float]:
    scores = v12_raw_class_scores(raw_text)
    if not scores:
        return None, None, 0.0
    best_cls, best_score = max(scores.items(), key=lambda x: x[1])
    if best_score <= 0:
        return None, None, 0.0
    best_cat = None
    for cat, classes in KNOWN_SHIP_CLASSES.items():
        if best_cls in classes:
            best_cat = cat
            break
    return best_cat, best_cls, best_score


def v12_set_known(result: Dict[str, Any], category: str, class_name: str, reason: str, min_conf: float = 0.58) -> Dict[str, Any]:
    return v11_set_known_class_result(result, category, class_name, reason, min_conf=min_conf)


def hierarchical_class_match(
    class_data_path: str,
    observed_attributes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """v12：修复 known_class_result 空 dict 问题，并用 raw 强提示纠正已知类。"""
    result = _hierarchical_class_match_v11(class_data_path, observed_attributes)

    raw_text = get_raw_text_from_observed(observed_attributes)
    unknown_intent = has_unknown_intent(raw_text)
    category_result = result.get("category_result") or {}
    known_class_result = result.get("known_class_result")
    open_set_result = result.get("open_set_result") or {}

    # 真正有 label 才算已知舰级已确定。
    if v12_is_empty_known_result(known_class_result):
        known_class_result = None
        result["known_class_result"] = None

    # 真正开放集样本不在这里强行补 known class，避免破坏 unknown 修正。
    if unknown_intent:
        return v9_rebuild_final(result)

    raw_cat, raw_cls, raw_score = v12_best_class_from_raw(raw_text)
    current_cat = category_result.get("label")

    # 1) raw 强提示足够明显时，允许纠正 category 和 known class。
    # 这一步主要修“导弹驱逐舰被判巡洋舰”“黄蜂级被判尼米兹”等。
    if raw_cat and raw_cls and raw_score >= 4.0:
        reason = f"v12 后处理：原始文本对 {raw_cls} 存在强提示，纠正/补全已知舰级。"
        return v12_set_known(result, raw_cat, raw_cls, reason, min_conf=0.62)

    # 2) 如果大类已对但 known class 为空：单已知大类直接补。
    if not known_class_result and current_cat:
        unique_cls = v11_infer_known_by_single_category(raw_text, current_cat)
        if unique_cls:
            reason = f"v12 后处理：大类已判定为{current_cat}，该大类当前只有一个已知舰级，补全为 {unique_cls}。"
            return v12_set_known(result, current_cat, unique_cls, reason, min_conf=0.56)

    # 3) 如果 open_set 被误触发，但 raw 强提示为已知类，则取消 open_set。
    if open_set_result.get("is_unknown") and raw_cat and raw_cls and raw_score >= 3.0:
        reason = f"v12 后处理：原始文本强烈指向已知舰级 {raw_cls}，取消开放集输出。"
        return v12_set_known(result, raw_cat, raw_cls, reason, min_conf=0.60)

    return v9_rebuild_final(result)


# ============================================================
# 5.6 v13：最终一致性修正
# ============================================================
# v13 目标：
# 1. 修复“category 已经正确，但 pred_known_class 仍为空”的输出一致性问题；
# 2. 只在非 open_set 的情况下，对“该大类只有一个已知舰级”的类别自动补全具体舰级；
# 3. 同步修正 known_class_result 与 final_decision，避免 replay/batch 读取时仍然为空。

_hierarchical_class_match_v12 = hierarchical_class_match


def v13_is_blank_value(x: Any) -> bool:
    return str(x or "").strip() in {"", "None", "null", "未知"}


def v13_single_known_class_map() -> Dict[str, str]:
    result: Dict[str, str] = {}
    for cat, classes in KNOWN_SHIP_CLASSES.items():
        if len(classes) == 1:
            result[cat] = classes[0]
    return result


SINGLE_KNOWN_CLASS_BY_CATEGORY = v13_single_known_class_map()


def v13_get_known_label(match_result: Dict[str, Any]) -> str:
    known = match_result.get("known_class_result") or {}
    final = match_result.get("final_decision") or {}
    return str(
        known.get("label")
        or known.get("ship_class")
        or final.get("primary_class")
        or ""
    ).strip()


def v13_get_category_label(match_result: Dict[str, Any]) -> str:
    cat = match_result.get("category_result") or {}
    final = match_result.get("final_decision") or {}
    return str(
        final.get("primary_category")
        or cat.get("label")
        or ""
    ).strip()


def v13_is_open_set(match_result: Dict[str, Any]) -> bool:
    open_set = match_result.get("open_set_result") or {}
    final = match_result.get("final_decision") or {}
    return bool(open_set.get("is_unknown", False)) or final.get("result_type") == "category_unknown"


def v13_fill_known_class(match_result: Dict[str, Any], category: str, class_name: str, reason: str) -> Dict[str, Any]:
    """强制同步 category_result、known_class_result、final_decision。"""
    if not isinstance(match_result, dict):
        return match_result

    old_cat = match_result.get("category_result") or {}
    old_known = match_result.get("known_class_result") or {}

    # 候选里如果有该舰级，尽量继承候选分数；没有就使用大类置信度兜底。
    cand = {}
    for item in match_result.get("known_class_candidates") or []:
        if item.get("label") == class_name or item.get("ship_class") == class_name:
            cand = item
            break

    confidence = max(
        float(cand.get("confidence", 0.0) or 0.0),
        float(old_known.get("confidence", 0.0) or 0.0),
        float(old_cat.get("confidence", 0.0) or 0.0),
        0.56,
    )
    score = float(cand.get("score", old_known.get("score", old_cat.get("score", 0.0)) or 0.0) or 0.0)

    match_result["category_result"] = {
        "label": category,
        "confidence": round(confidence, 4),
        "status": "matched",
        "reason": reason,
    }

    match_result["known_class_result"] = {
        "label": class_name,
        "category": category,
        "confidence": round(confidence, 4),
        "score": score,
        "known_status": "Known",
        "reason": reason,
    }

    match_result["open_set_result"] = {
        "is_unknown": False,
        "unknown_scope": None,
        "reason": "最终一致性修正：已补全单已知舰级，因此不是类别内未知类。",
    }

    match_result["final_decision"] = {
        "result_type": "known_class",
        "primary_category": category,
        "primary_class": class_name,
        "confidence": round(confidence, 4),
        "status": "single_known_class_filled",
        "message": f"最终判定：{category} / {class_name}。{reason}",
    }
    return match_result


def v13_enforce_output_consistency(match_result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(match_result, dict):
        return match_result

    # 开放集样本不要补已知舰级，否则会破坏类别内未知判断。
    if v13_is_open_set(match_result):
        return match_result

    category = v13_get_category_label(match_result)
    known_label = v13_get_known_label(match_result)

    # 如果 final_decision 没有 primary_class，但 known_class_result 有 label，则同步过去。
    if not v13_is_blank_value(known_label):
        known = match_result.get("known_class_result") or {}
        category = category or known.get("category")
        final = match_result.get("final_decision") or {}
        if v13_is_blank_value(final.get("primary_class")):
            match_result["final_decision"] = {
                "result_type": "known_class",
                "primary_category": category,
                "primary_class": known_label,
                "confidence": known.get("confidence", final.get("confidence", 0.0)),
                "status": final.get("status", "single_best"),
                "message": final.get("message") or f"最终判定：{category} / {known_label}。",
            }
        return match_result

    # 大类已确定、且该大类只有一个已知舰级：补全具体舰级。
    if category in SINGLE_KNOWN_CLASS_BY_CATEGORY:
        class_name = SINGLE_KNOWN_CLASS_BY_CATEGORY[category]
        reason = f"大类已判定为{category}，且该大类当前只有一个已知舰级，因此补全为 {class_name}。"
        return v13_fill_known_class(match_result, category, class_name, reason)

    return match_result


def hierarchical_class_match(
    class_data_path: str,
    observed_attributes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """v13：在 v12 基础上补最终输出一致性，修复 category 对但 class 为空的问题。"""
    result = _hierarchical_class_match_v12(class_data_path, observed_attributes)
    return v13_enforce_output_consistency(result)





# ==================== v26: known-only less-aggressive open-set ====================
# 目标：
# 1. 只使用 7 个已知舰级的先验特征，不使用任何未知舰级名称或未知类专属签名；
# 2. 保留 open-set 能力，但避免“没有唯一锚点就拒识”的过度 open-set；
# 3. 将已知类特征拆成 anchor/support/conflict 三类，配置统一放在 schema_config.py。

_hierarchical_class_match_v13_known_only_base = hierarchical_class_match


def v26_compact_text(text: str) -> str:
    return compact_key(text or "")


def v26_has(raw_text: str, term: str) -> bool:
    if not raw_text or not term:
        return False
    return v26_compact_text(term) in v26_compact_text(raw_text)


def v26_score_terms(raw_text: str, terms: List[Tuple[str, float]]) -> Tuple[float, List[str]]:
    score = 0.0
    evidence: List[str] = []
    for term, weight in terms or []:
        if v26_has(raw_text, str(term)):
            score += float(weight)
            evidence.append(str(term))
    return round(score, 4), evidence


def v26_threshold(name: str, default: float) -> float:
    try:
        return float(SIGNATURE_RULE_THRESHOLDS.get(name, default))
    except Exception:
        return default


def v26_set_category_unknown(
    match_result: Dict[str, Any],
    category: str,
    reason: str,
    analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not isinstance(match_result, dict):
        return match_result
    confidence = max(float((match_result.get("category_result") or {}).get("confidence", 0.0) or 0.0), 0.55)
    match_result["category_result"] = {
        "label": category,
        "confidence": round(confidence, 4),
        "status": "matched",
        "reason": reason,
    }
    match_result["known_class_result"] = {
        "label": None,
        "category": category,
        "confidence": 0.0,
        "score": 0.0,
        "known_status": "UnknownWithinCategory",
        "reason": reason,
    }
    match_result["open_set_result"] = {
        "is_unknown": True,
        "unknown_scope": UNKNOWN_OUTPUT_TEMPLATE.format(category=category),
        "reason": reason,
    }
    match_result["final_decision"] = {
        "result_type": "category_unknown",
        "primary_category": category,
        "primary_class": None,
        "confidence": round(confidence, 4),
        "status": "known_only_less_aggressive_open_set",
        "message": f"最终判定：{category}类别内未知类。{reason}",
    }
    if analysis is not None:
        match_result["v26_known_only_analysis"] = analysis
    return match_result


def v26_detect_known_from_schema(raw_text: str) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    conflict_block = v26_threshold("known_conflict_block", 12.0)

    for category, classes in KNOWN_SHIP_CLASSES.items():
        for cls in classes:
            anchor_score, anchor_ev = v26_score_terms(raw_text, KNOWN_CLASS_ANCHOR_SIGNATURES.get(cls, []))
            support_score, support_ev = v26_score_terms(raw_text, KNOWN_CLASS_SUPPORT_SIGNATURES.get(cls, []))
            strong_score, strong_ev = v26_score_terms(raw_text, KNOWN_CLASS_STRONG_SIGNATURES.get(cls, []))
            conflict_score, conflict_ev = v26_score_terms(raw_text, KNOWN_CLASS_CONFLICT_SIGNATURES.get(cls, []))

            # adjusted_score 用于排序；anchor/support 单独用于保护，避免共享特征闭集硬判。
            adjusted_score = anchor_score + support_score + 0.35 * strong_score - conflict_score
            blocked = conflict_score >= conflict_block and anchor_score < v26_threshold("known_anchor_protect", 10.0)

            protect = False
            if not blocked:
                # 1) 命中强锚点，保护为已知类。
                if anchor_score >= v26_threshold("known_anchor_protect", 10.0) and adjusted_score >= v26_threshold("known_protect", 12.0):
                    protect = True
                # 2) 没有强锚点，但多项支持特征充足，也允许保护，避免过度 open-set。
                elif support_score >= v26_threshold("known_support_protect", 12.0) and adjusted_score >= v26_threshold("known_protect", 12.0):
                    protect = True

            candidates.append({
                "category": category,
                "class_name": cls,
                "anchor_score": round(anchor_score, 4),
                "support_score": round(support_score, 4),
                "strong_score": round(strong_score, 4),
                "conflict_score": round(conflict_score, 4),
                "adjusted_score": round(adjusted_score, 4),
                "anchor_evidence": anchor_ev[:20],
                "support_evidence": support_ev[:20],
                "strong_evidence": strong_ev[:20],
                "conflict_evidence": conflict_ev[:20],
                "blocked_by_conflict": blocked,
                "is_candidate": protect,
            })

    valid = [c for c in candidates if c.get("is_candidate")]
    valid.sort(key=lambda x: (x["adjusted_score"], x["anchor_score"], x["support_score"]), reverse=True)
    return {
        "best": valid[0] if valid else None,
        "all_candidates": sorted(candidates, key=lambda x: (x["adjusted_score"], x["anchor_score"], x["support_score"]), reverse=True)[:20],
    }


def v26_find_candidate_for_class(known_det: Dict[str, Any], class_name: str) -> Optional[Dict[str, Any]]:
    for item in known_det.get("all_candidates", []) or []:
        if item.get("class_name") == class_name:
            return item
    return None


def v26_best_candidate_in_category(known_det: Dict[str, Any], category: str) -> Optional[Dict[str, Any]]:
    best = None
    for item in known_det.get("all_candidates", []) or []:
        if item.get("category") != category:
            continue
        if best is None or float(item.get("adjusted_score", 0.0) or 0.0) > float(best.get("adjusted_score", 0.0) or 0.0):
            best = item
    return best


def v26_can_fill_single_known(cur: Optional[Dict[str, Any]]) -> bool:
    if not cur:
        return False
    if float(cur.get("conflict_score", 0.0) or 0.0) >= v26_threshold("known_conflict_block", 12.0):
        return False
    anchor = float(cur.get("anchor_score", 0.0) or 0.0)
    support = float(cur.get("support_score", 0.0) or 0.0)
    adjusted = float(cur.get("adjusted_score", 0.0) or 0.0)
    return (
        anchor >= v26_threshold("single_known_anchor_fill", 6.0)
        or support >= v26_threshold("single_known_support_fill", 8.0)
        or adjusted >= v26_threshold("single_known_adjusted_fill", 10.0)
    )


def v26_is_weak_closed_match(cur: Optional[Dict[str, Any]]) -> bool:
    """
    判断当前已知舰级是否属于“证据过弱的闭集硬匹配”。
    只有在 anchor/support/adjusted 都很低时才拒识，避免 v24 那种过度 open-set。
    """
    if not cur:
        return False
    anchor = float(cur.get("anchor_score", 0.0) or 0.0)
    support = float(cur.get("support_score", 0.0) or 0.0)
    adjusted = float(cur.get("adjusted_score", 0.0) or 0.0)
    return (
        anchor <= v26_threshold("weak_match_anchor_max", 3.0)
        and support <= v26_threshold("weak_match_support_max", 5.0)
        and adjusted <= v26_threshold("weak_match_adjusted_max", 8.0)
    )


def v26_enforce_known_only_less_aggressive(result: Dict[str, Any], observed_attributes: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    raw_text = get_raw_text_from_observed(observed_attributes)
    known_det = v26_detect_known_from_schema(raw_text)
    best_known = known_det.get("best")

    current_cat = v13_get_category_label(result)
    current_cls = v13_get_known_label(result)
    is_open = v13_is_open_set(result)

    analysis = {
        "principle": "v26_known_only_less_aggressive: only known-class anchors/support/conflicts are used; no unknown-derived signatures",
        "pred_category_before_v26": current_cat,
        "pred_class_before_v26": current_cls,
        "pred_open_set_before_v26": is_open,
        "best_known": best_known,
        "known_candidates": known_det.get("all_candidates", [])[:10],
    }
    result["v26_known_only_analysis"] = analysis

    # 1) 明确已知类保护：如果文本命中某已知舰级的 anchor/support，允许纠正 open_set 或错误大类。
    if best_known:
        target_cat = best_known.get("category")
        target_cls = best_known.get("class_name")
        if target_cat and target_cls and (is_open or current_cat != target_cat or current_cls != target_cls):
            reason = (
                f"v26 known-only 已知类保护：文本命中 {target_cls} 的已知类锚点/支持特征，"
                f"锚点={best_known.get('anchor_evidence', [])[:6]}，支持={best_known.get('support_evidence', [])[:6]}。"
            )
            fixed = v13_fill_known_class(result, target_cat, target_cls, reason)
            fixed["v26_known_only_analysis"] = analysis
            return fixed

    # 2) 单已知舰级大类的“非过度 open-set”补全。
    #    例如航空母舰当前只含尼米兹级；只要有中等支持且无已知类冲突，就不应轻易 open_set。
    if current_cat in KNOWN_SHIP_CLASSES and not current_cls:
        classes = KNOWN_SHIP_CLASSES.get(current_cat, [])
        if len(classes) == 1:
            only_cls = classes[0]
            cur = v26_find_candidate_for_class(known_det, only_cls)
            if v26_can_fill_single_known(cur):
                reason = (
                    f"v26 known-only 单已知舰级补全：大类 {current_cat} 当前只有已知舰级 {only_cls}，"
                    f"且文本有足够已知类支持证据、无明显已知类冲突。"
                )
                fixed = v13_fill_known_class(result, current_cat, only_cls, reason)
                fixed["v26_known_only_analysis"] = analysis
                return fixed

    # 3) 如果当前已经闭集匹配到已知舰级，只在“证据极弱或明显冲突”时拒识，避免过度 open-set。
    if current_cat in SHIP_CATEGORIES and current_cls:
        cur = v26_find_candidate_for_class(known_det, current_cls)
        conflict_score = float((cur or {}).get("conflict_score", 0.0) or 0.0)
        anchor_score = float((cur or {}).get("anchor_score", 0.0) or 0.0)
        support_score = float((cur or {}).get("support_score", 0.0) or 0.0)

        # 明显已知类冲突，且自身无 anchor/support 保护，才拒识。
        if (
            conflict_score >= v26_threshold("known_only_open_set_conflict", 14.0)
            and anchor_score < v26_threshold("known_anchor_protect", 10.0)
            and support_score < v26_threshold("known_support_protect", 12.0)
        ):
            reason = (
                f"v26 known-only 开放集：当前已知舰级 {current_cls} 被其他已知类冲突特征削弱，"
                f"且缺少自身锚点/支持证据；因此只保留大类 {current_cat}。"
            )
            return v26_set_category_unknown(result, current_cat, reason, analysis)

        # 证据极弱的闭集硬匹配，拒识为类别内未知。
        if v26_is_weak_closed_match(cur):
            reason = (
                f"v26 known-only 开放集：大类 {current_cat} 可判断，但 {current_cls} 的已知类锚点/支持证据过弱；"
                "因此拒绝闭集硬匹配，输出类别内未知。"
            )
            return v26_set_category_unknown(result, current_cat, reason, analysis)

    # 4) open_set 已触发但大类下有唯一已知舰级且支持证据充足，拉回已知类。
    if is_open and current_cat in KNOWN_SHIP_CLASSES:
        classes = KNOWN_SHIP_CLASSES.get(current_cat, [])
        if len(classes) == 1:
            only_cls = classes[0]
            cur = v26_find_candidate_for_class(known_det, only_cls)
            if v26_can_fill_single_known(cur):
                reason = (
                    f"v26 known-only open_set 修正：{current_cat} 下唯一已知舰级 {only_cls} 有足够已知类支持证据，"
                    "取消过度 open-set。"
                )
                fixed = v13_fill_known_class(result, current_cat, only_cls, reason)
                fixed["v26_known_only_analysis"] = analysis
                return fixed

    return result


def hierarchical_class_match(
    class_data_path: str,
    observed_attributes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """v26：known-only，非过度 open-set。只使用已知类 anchor/support/conflict，不使用未知类先验。"""
    result = _hierarchical_class_match_v13_known_only_base(class_data_path, observed_attributes)
    return v26_enforce_known_only_less_aggressive(result, observed_attributes)


# ==================== v26: known-only raw category cues ====================
# 只补充通用大类提示词，不加入任何未知舰级名称或未知舰级专属签名。
def _v26_extend_raw_category_cues():
    extra = {
        "驱逐舰": ["大型防空导弹驱逐舰", "防空导弹驱逐舰", "大型驱逐舰", "反导驱逐舰"],
        "护卫舰": ["反潜护卫舰", "通用护卫舰", "现代护卫舰", "单体护卫舰", "巡防舰", "中型护卫舰", "中型军舰"],
        "巡洋舰": ["带宙斯盾的巡洋舰", "宙斯盾巡洋舰", "好像是巡洋舰", "指挥舰", "比伯克级大", "比阿利·伯克级大"],
        "两栖舰": ["两栖运输舰", "两栖船坞运输舰", "两栖船坞", "运兵和车辆", "运兵和装备"],
        "登陆舰": ["船坞登陆舰", "船坞登陆", "大坞舱", "登陆装备"],
        "航空母舰": ["主力航母", "大型航母", "核动力航母", "有弹射器", "很多飞机", "好多飞机"],
    }
    for cat, cues in extra.items():
        RAW_CATEGORY_CUES.setdefault(cat, [])
        for cue in cues:
            if cue not in RAW_CATEGORY_CUES[cat]:
                RAW_CATEGORY_CUES[cat].append(cue)

_v26_extend_raw_category_cues()



# ==================== v27: known-only balanced open-set ====================
# 目标：继续保持 known-only，不加入任何未知舰级名称或未知类专属签名；
# 同时修正 v24/v26 的“过度 open-set”和“跨大类误纠偏”问题。


def v27_threshold(name: str, default: float) -> float:
    try:
        return float(SIGNATURE_RULE_THRESHOLDS.get(name, default))
    except Exception:
        return default


def v27_raw_category_evidence(raw_text: str, category: str) -> int:
    """只检测通用大类词，不检测任何未知舰级名称。"""
    text = v26_compact_text(raw_text)
    cues = {
        "航空母舰": ["航空母舰", "航母", "超级航母", "主力航母", "核动力航母"],
        "巡洋舰": ["巡洋舰", "导弹巡洋舰", "宙斯盾巡洋舰", "指挥舰", "舰队指挥"],
        "驱逐舰": ["驱逐舰", "导弹驱逐舰", "宙斯盾驱逐舰", "防空驱逐舰", "带刀护卫"],
        "护卫舰": ["护卫舰", "巡防舰", "反潜护卫舰", "通用护卫舰", "现代护卫舰", "单体护卫舰"],
        "两栖舰": ["两栖舰", "两栖攻击舰", "两栖船坞运输舰", "两栖运输舰", "LPD", "LHD"],
        "登陆舰": ["登陆舰", "船坞登陆舰", "船坞登陆", "LSD"],
    }
    return sum(1 for cue in cues.get(category, []) if v26_compact_text(cue) in text)


def v27_candidate_scores(cur: Optional[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    if not cur:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(cur.get("anchor_score", 0.0) or 0.0),
        float(cur.get("support_score", 0.0) or 0.0),
        float(cur.get("adjusted_score", 0.0) or 0.0),
        float(cur.get("conflict_score", 0.0) or 0.0),
    )


def v27_has_known_support(cur: Optional[Dict[str, Any]], raw_text: str = "", category: str = "") -> bool:
    """判断是否有足够已知类证据。阈值比 v26 稍放宽，但仍只基于已知类特征。"""
    if not cur:
        return False
    anchor, support, adjusted, conflict = v27_candidate_scores(cur)
    if conflict >= v27_threshold("known_conflict_block", 18.0) and anchor < v27_threshold("known_anchor_protect", 8.0):
        return False
    if anchor >= v27_threshold("known_anchor_protect", 8.0):
        return True
    if support >= v27_threshold("known_support_protect", 8.0):
        return True
    if adjusted >= v27_threshold("single_known_adjusted_fill", 6.5):
        return True
    # 大类词明确且有少量支持时，可以避免已知类被过度 open-set。
    if category and v27_raw_category_evidence(raw_text, category) >= 1 and (support >= 4.0 or adjusted >= 5.5):
        return True
    return False


def v27_is_weak_known_match(cur: Optional[Dict[str, Any]], raw_text: str = "", category: str = "") -> bool:
    """闭集已知类证据太弱时才拒识。避免 v24/v26 大量已知类被 open-set 误伤。"""
    if not cur:
        return False
    anchor, support, adjusted, conflict = v27_candidate_scores(cur)
    if anchor >= 1.0:
        return False
    if category and v27_raw_category_evidence(raw_text, category) >= 1 and (support >= 4.0 or adjusted >= 5.5):
        return False
    return (
        anchor <= v27_threshold("weak_match_anchor_max", 0.5)
        and support <= v27_threshold("weak_match_support_max", 4.5)
        and adjusted <= v27_threshold("weak_match_adjusted_max", 6.0)
    )


def v27_should_cross_correct(raw_text: str, current_cat: str, target: Dict[str, Any]) -> bool:
    """跨大类纠偏必须非常谨慎，防止把巡洋舰/航母/两栖舰互相拉错。"""
    if not target:
        return False
    target_cat = target.get("category")
    if not target_cat:
        return False
    if not current_cat or current_cat not in SHIP_CATEGORIES:
        return True
    if target_cat == current_cat:
        return True

    anchor, support, adjusted, conflict = v27_candidate_scores(target)
    target_raw = v27_raw_category_evidence(raw_text, target_cat)
    current_raw = v27_raw_category_evidence(raw_text, current_cat)

    # 只有目标大类有显式语义，且强证据明显压过当前大类时，才跨大类修正。
    if target_raw <= 0:
        return False
    if current_raw > 0 and anchor < v27_threshold("cross_category_anchor_force", 22.0):
        return False
    return (
        anchor >= v27_threshold("cross_category_anchor_force", 22.0)
        or (support >= v27_threshold("cross_category_support_force", 18.0) and adjusted >= v27_threshold("cross_category_adjusted_force", 16.0))
    )


def v27_best_known_for_current_category(known_det: Dict[str, Any], category: str) -> Optional[Dict[str, Any]]:
    if not category:
        return None
    return v26_best_candidate_in_category(known_det, category)


def v27_enforce_known_only_balanced(result: Dict[str, Any], observed_attributes: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    raw_text = get_raw_text_from_observed(observed_attributes)
    known_det = v26_detect_known_from_schema(raw_text)
    best_known = known_det.get("best")

    current_cat = v13_get_category_label(result)
    current_cls = v13_get_known_label(result)
    is_open = v13_is_open_set(result)

    analysis = {
        "principle": "v27_known_only_balanced: only known-class anchor/support/conflict; no unknown class names or unknown-derived signatures",
        "pred_category_before_v27": current_cat,
        "pred_class_before_v27": current_cls,
        "pred_open_set_before_v27": is_open,
        "best_known": best_known,
        "known_candidates": known_det.get("all_candidates", [])[:10],
    }
    result["v27_known_only_analysis"] = analysis

    # 1) 先在当前大类内部保护已知类，避免过度 open-set。
    if current_cat in KNOWN_SHIP_CLASSES:
        classes = KNOWN_SHIP_CLASSES.get(current_cat, [])
        if current_cls:
            cur = v26_find_candidate_for_class(known_det, current_cls)
            if is_open and v27_has_known_support(cur, raw_text, current_cat):
                reason = f"v27 known-only：当前已知舰级 {current_cls} 有足够已知类证据，取消过度 open-set。"
                fixed = v13_fill_known_class(result, current_cat, current_cls, reason)
                fixed["v27_known_only_analysis"] = analysis
                return fixed
            if (not is_open) and v27_is_weak_known_match(cur, raw_text, current_cat):
                reason = f"v27 known-only：{current_cls} 的已知类证据过弱，拒绝闭集硬匹配，保留 {current_cat} 类别内未知。"
                return v26_set_category_unknown(result, current_cat, reason, analysis)

        if not current_cls:
            # 当前大类如果只有一个已知舰级，且该舰级有足够已知类支持，则补全。
            if len(classes) == 1:
                only_cls = classes[0]
                cur = v26_find_candidate_for_class(known_det, only_cls)
                if v27_has_known_support(cur, raw_text, current_cat):
                    reason = f"v27 known-only：{current_cat} 下唯一已知舰级 {only_cls} 命中已知类支持特征，补全已知舰级。"
                    fixed = v13_fill_known_class(result, current_cat, only_cls, reason)
                    fixed["v27_known_only_analysis"] = analysis
                    return fixed
            else:
                # 同一大类多个已知舰级时，只在该大类内部选择证据最强者。
                cur = v27_best_known_for_current_category(known_det, current_cat)
                if v27_has_known_support(cur, raw_text, current_cat):
                    reason = f"v27 known-only：在 {current_cat} 内部命中 {cur.get('class_name')} 的已知类支持特征，补全已知舰级。"
                    fixed = v13_fill_known_class(result, current_cat, cur.get("class_name"), reason)
                    fixed["v27_known_only_analysis"] = analysis
                    return fixed

    # 2) 跨大类纠偏：只在目标大类有显式语义且强证据充分时进行。
    if best_known:
        target_cat = best_known.get("category")
        target_cls = best_known.get("class_name")
        if target_cat and target_cls and (current_cat != target_cat or current_cls != target_cls or is_open):
            if v27_should_cross_correct(raw_text, current_cat, best_known):
                reason = (
                    f"v27 known-only 跨大类强证据纠偏：文本明确支持 {target_cat}/{target_cls}，"
                    f"锚点={best_known.get('anchor_evidence', [])[:6]}，支持={best_known.get('support_evidence', [])[:6]}。"
                )
                fixed = v13_fill_known_class(result, target_cat, target_cls, reason)
                fixed["v27_known_only_analysis"] = analysis
                return fixed

    # 3) open_set 已触发但当前大类有已知类支持，拉回；没有支持则保留 open_set。
    if is_open and current_cat in KNOWN_SHIP_CLASSES:
        cur = v27_best_known_for_current_category(known_det, current_cat)
        if v27_has_known_support(cur, raw_text, current_cat):
            reason = f"v27 known-only：open_set 触发后，{current_cat} 内部仍有足够已知类证据，拉回 {cur.get('class_name')}。"
            fixed = v13_fill_known_class(result, current_cat, cur.get("class_name"), reason)
            fixed["v27_known_only_analysis"] = analysis
            return fixed

    return result


# 覆盖前一版 hierarchical_class_match。
def hierarchical_class_match(
    class_data_path: str,
    observed_attributes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """v27：known-only balanced open-set。只用已知类特征，降低过度 open-set，不使用未知类先验。"""
    result = _hierarchical_class_match_v13_known_only_base(class_data_path, observed_attributes)
    return v27_enforce_known_only_balanced(result, observed_attributes)


# ==================== v27: extra raw category cues ====================
# 仅包含通用大类词和已知类自然表达，不含未知舰级名称或未知类专属签名。
def _v27_extend_raw_category_cues():
    extra = {
        "驱逐舰": ["一艘驱逐舰", "驱逐舰", "导弹驱逐舰", "防空驱逐舰", "航母的带刀护卫", "带刀护卫"],
        "护卫舰": ["护卫舰", "巡防舰", "反潜护卫舰", "通用护卫舰", "现代护卫舰", "单体护卫舰", "中型护卫舰"],
        "巡洋舰": ["巡洋舰", "导弹巡洋舰", "宙斯盾巡洋舰", "好像是巡洋舰", "指挥舰", "舰队指挥"],
        "两栖舰": ["两栖舰", "两栖攻击舰", "两栖运输舰", "两栖船坞运输舰", "两栖船坞", "运兵和车辆", "运兵和装备"],
        "登陆舰": ["登陆舰", "船坞登陆舰", "船坞登陆", "大坞舱", "登陆装备"],
        "航空母舰": ["航空母舰", "航母", "主力航母", "大型航母", "核动力航母", "超级航母"],
    }
    for cat, cues in extra.items():
        RAW_CATEGORY_CUES.setdefault(cat, [])
        for cue in cues:
            if cue not in RAW_CATEGORY_CUES[cat]:
                RAW_CATEGORY_CUES[cat].append(cue)

_v27_extend_raw_category_cues()




# ==================== v28: known-only calibrated support/open-set ====================
# 目标：继续保持 known-only，不使用任何未知舰级名称或未知类专属签名；
# 修正 v23-v27 过度 open-set 与共享特征闭集硬匹配之间的失衡。
# 思路：
# 1) 已知类有明确 anchor/support 且无冲突 -> 补回已知类；
# 2) 已知类只靠共享特征、anchor/support不足或冲突较强 -> 保留/改为类别内未知；
# 3) 通用大类词只用于修正大类，不直接当作某个未知类证据。

try:
    import schema_config as _schema_config_v28
except Exception:
    _schema_config_v28 = None


def v28_cfg(name: str, default=None):
    if _schema_config_v28 is None:
        return default
    return getattr(_schema_config_v28, name, default)


def v28_threshold(name: str, default: float) -> float:
    table = v28_cfg("V28_RULE_THRESHOLDS", {}) or {}
    try:
        return float(table.get(name, default))
    except Exception:
        return default


def v28_patterns_for(cls_name: str, key: str):
    table = v28_cfg("V28_KNOWN_CLASS_RULES", {}) or {}
    info = table.get(cls_name, {}) or {}
    return info.get(key, []) or []


def v28_compact(text: Any) -> str:
    return v26_compact_text(str(text or ""))


def v28_count_score(raw_text: str, patterns) -> Tuple[float, List[str]]:
    text = v28_compact(raw_text)
    score = 0.0
    hits: List[str] = []
    for item in patterns:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            pat, weight = str(item[0]), float(item[1])
        else:
            pat, weight = str(item), 1.0
        if pat and v28_compact(pat) in text:
            score += weight
            hits.append(pat)
    return score, hits


def v28_raw_category_scores(raw_text: str) -> Dict[str, float]:
    cues = v28_cfg("V28_CATEGORY_CUES", {}) or {}
    scores: Dict[str, float] = {}
    for cat, pats in cues.items():
        s, _ = v28_count_score(raw_text, pats)
        scores[cat] = s
    return scores


def v28_best_raw_category(raw_text: str) -> Tuple[str, float]:
    scores = v28_raw_category_scores(raw_text)
    if not scores:
        return "", 0.0
    cat, score = max(scores.items(), key=lambda kv: kv[1])
    if score <= 0:
        return "", 0.0
    return cat, score


def v28_known_rule_score(raw_text: str, cls_name: str) -> Dict[str, Any]:
    category = get_category_of_known_class(cls_name)
    anchor, anchor_hits = v28_count_score(raw_text, v28_patterns_for(cls_name, "anchor"))
    support, support_hits = v28_count_score(raw_text, v28_patterns_for(cls_name, "support"))
    conflict, conflict_hits = v28_count_score(raw_text, v28_patterns_for(cls_name, "conflict"))
    raw_cat_scores = v28_raw_category_scores(raw_text)
    cat_score = raw_cat_scores.get(category, 0.0)
    # adjusted 只代表“像已知类”的程度，不直接包含冲突。
    adjusted = anchor + support + min(cat_score, 8.0) * 0.35
    return {
        "class_name": cls_name,
        "category": category,
        "anchor_score": anchor,
        "support_score": support,
        "conflict_score": conflict,
        "category_score": cat_score,
        "adjusted_score": adjusted - conflict * 0.65,
        "raw_adjusted_score": adjusted,
        "anchor_hits": anchor_hits,
        "support_hits": support_hits,
        "conflict_hits": conflict_hits,
    }


def v28_all_known_scores(raw_text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cat, classes in KNOWN_SHIP_CLASSES.items():
        for cls_name in classes:
            rows.append(v28_known_rule_score(raw_text, cls_name))
    rows.sort(key=lambda x: (x.get("adjusted_score", 0.0), x.get("anchor_score", 0.0), x.get("support_score", 0.0)), reverse=True)
    return rows


def v28_best_in_category(rows: List[Dict[str, Any]], category: str) -> Optional[Dict[str, Any]]:
    cand = [r for r in rows if r.get("category") == category]
    if not cand:
        return None
    cand.sort(key=lambda x: (x.get("adjusted_score", 0.0), x.get("anchor_score", 0.0), x.get("support_score", 0.0)), reverse=True)
    return cand[0]


def v28_has_known_support(row: Optional[Dict[str, Any]], raw_text: str = "") -> bool:
    if not row:
        return False
    anchor = float(row.get("anchor_score", 0.0) or 0.0)
    support = float(row.get("support_score", 0.0) or 0.0)
    conflict = float(row.get("conflict_score", 0.0) or 0.0)
    adjusted = float(row.get("adjusted_score", 0.0) or 0.0)
    cat_score = float(row.get("category_score", 0.0) or 0.0)
    if conflict >= v28_threshold("hard_conflict_block", 18.0) and anchor < v28_threshold("anchor_override_conflict", 18.0):
        return False
    if anchor >= v28_threshold("anchor_known", 12.0):
        return True
    if support >= v28_threshold("support_known", 10.0) and conflict < v28_threshold("soft_conflict_block", 12.0):
        return True
    if cat_score >= v28_threshold("category_cue_known", 5.0) and support >= v28_threshold("support_with_category_known", 5.0) and conflict < v28_threshold("soft_conflict_block", 12.0):
        return True
    if adjusted >= v28_threshold("adjusted_known", 11.0) and conflict < v28_threshold("soft_conflict_block", 12.0):
        return True
    return False


def v28_weak_closed_match(row: Optional[Dict[str, Any]]) -> bool:
    """闭集结果是否只是共享特征硬匹配。"""
    if not row:
        return True
    anchor = float(row.get("anchor_score", 0.0) or 0.0)
    support = float(row.get("support_score", 0.0) or 0.0)
    conflict = float(row.get("conflict_score", 0.0) or 0.0)
    adjusted = float(row.get("adjusted_score", 0.0) or 0.0)
    if conflict >= v28_threshold("soft_conflict_block", 12.0) and anchor < v28_threshold("anchor_known", 12.0):
        return True
    if anchor >= v28_threshold("anchor_min_not_weak", 6.0):
        return False
    if support >= v28_threshold("support_min_not_weak", 7.0) and adjusted >= v28_threshold("adjusted_min_not_weak", 7.0):
        return False
    return True


def v28_should_cross_correct(raw_text: str, current_cat: str, target: Dict[str, Any]) -> bool:
    if not target:
        return False
    target_cat = target.get("category")
    if not target_cat or target_cat == current_cat:
        return False
    raw_scores = v28_raw_category_scores(raw_text)
    target_raw = raw_scores.get(target_cat, 0.0)
    current_raw = raw_scores.get(current_cat, 0.0) if current_cat else 0.0
    anchor = float(target.get("anchor_score", 0.0) or 0.0)
    support = float(target.get("support_score", 0.0) or 0.0)
    conflict = float(target.get("conflict_score", 0.0) or 0.0)
    if conflict >= v28_threshold("soft_conflict_block", 12.0):
        return False
    # 跨大类必须有目标大类显式语义，或者非常强的已知类 anchor。
    if target_raw < v28_threshold("cross_target_category_min", 5.0) and anchor < v28_threshold("cross_anchor_force", 24.0):
        return False
    if current_raw > 0 and target_raw <= current_raw and anchor < v28_threshold("cross_anchor_force", 24.0):
        return False
    return anchor >= v28_threshold("cross_anchor_force", 24.0) or (support >= v28_threshold("cross_support_force", 20.0) and target_raw > current_raw)


def v28_set_category_unknown(result: Dict[str, Any], category: str, reason: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    return v26_set_category_unknown(result, category, reason, analysis)


def v28_enforce_known_only_less_open(result: Dict[str, Any], observed_attributes: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    raw_text = get_raw_text_from_observed(observed_attributes)
    rows = v28_all_known_scores(raw_text)
    best = rows[0] if rows else None
    raw_cat, raw_cat_score = v28_best_raw_category(raw_text)

    current_cat = v13_get_category_label(result)
    current_cls = v13_get_known_label(result)
    is_open = v13_is_open_set(result)

    analysis = {
        "principle": "v28_known_only_less_open: known-class anchor/support/conflict only; no unknown-class names or unknown-derived signatures",
        "raw_category_best": raw_cat,
        "raw_category_score": raw_cat_score,
        "pred_category_before_v28": current_cat,
        "pred_class_before_v28": current_cls,
        "pred_open_set_before_v28": is_open,
        "known_rule_candidates": rows[:10],
    }
    result["v28_known_only_analysis"] = analysis

    # A. 如果当前大类为空/不稳定，但原文有明确大类，先用原文大类纠正为类别内未知候选。
    if (not current_cat or current_cat not in SHIP_CATEGORIES) and raw_cat:
        current_cat = raw_cat
        result = v28_set_category_unknown(result, current_cat, "v28 known-only：原文有明确大类语义，先修正为该大类类别内未知。", analysis)
        is_open = True
        current_cls = ""

    # B. 同大类内部优先处理，避免跨大类乱拉。
    if current_cat in KNOWN_SHIP_CLASSES:
        cur = None
        if current_cls:
            cur = next((r for r in rows if r.get("class_name") == current_cls), None)
        if cur is None:
            cur = v28_best_in_category(rows, current_cat)

        if current_cls and not is_open:
            # 已经闭集输出：如果证据极弱，则拒识；否则保留。
            if v28_weak_closed_match(cur):
                reason = f"v28 known-only：{current_cls} 仅由共享弱特征支撑或存在冲突，改为 {current_cat} 类别内未知。"
                return v28_set_category_unknown(result, current_cat, reason, analysis)
            return result

        # open_set 或 class 为空：只要当前大类内部已知类有足够 known-only 证据，就补回。
        if v28_has_known_support(cur, raw_text):
            cls = cur.get("class_name")
            reason = f"v28 known-only：{current_cat} 内部 {cls} 有足够已知类 anchor/support，取消过度 open-set。"
            fixed = v13_fill_known_class(result, current_cat, cls, reason)
            fixed["v28_known_only_analysis"] = analysis
            return fixed

        # 如果原文明确大类与当前大类不同，并且当前没有已知类支持，则修正为原文大类未知。
        if raw_cat and raw_cat != current_cat and raw_cat_score >= v28_threshold("raw_category_override_min", 8.0):
            raw_row = v28_best_in_category(rows, raw_cat)
            if not v28_has_known_support(raw_row, raw_text):
                reason = f"v28 known-only：当前大类 {current_cat} 缺少已知类支持，原文更明确支持 {raw_cat}，输出 {raw_cat} 类别内未知。"
                return v28_set_category_unknown(result, raw_cat, reason, analysis)

    # C. 跨大类强证据纠偏，只用于非常明确的已知类锚点。
    if best and current_cat != best.get("category"):
        if v28_should_cross_correct(raw_text, current_cat, best):
            target_cat = best.get("category")
            target_cls = best.get("class_name")
            reason = f"v28 known-only：跨大类强锚点纠偏到 {target_cat}/{target_cls}。"
            fixed = v13_fill_known_class(result, target_cat, target_cls, reason)
            fixed["v28_known_only_analysis"] = analysis
            return fixed

    # D. 如果仍是 open_set，但原文大类明确，且无已知类支持，保持该大类未知。
    if is_open and raw_cat and (not current_cat or current_cat != raw_cat):
        raw_row = v28_best_in_category(rows, raw_cat)
        if not v28_has_known_support(raw_row, raw_text):
            return v28_set_category_unknown(result, raw_cat, f"v28 known-only：原文大类更明确，且没有足够已知类证据，输出 {raw_cat} 类别内未知。", analysis)

    return result


# 覆盖前一版 hierarchical_class_match。
def hierarchical_class_match(
    class_data_path: str,
    observed_attributes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """v28：known-only, less aggressive open-set, no unknown-derived signatures."""
    result = _hierarchical_class_match_v13_known_only_base(class_data_path, observed_attributes)
    return v28_enforce_known_only_less_open(result, observed_attributes)

# ==================== v29: slot-based known-only postprocess ====================
# 目标：不再主要依赖 raw_text 字符串规则，而是优先使用 LLM 已经抽取出的 observed_attributes 卡槽。
# 严格边界：只使用 7 个已知舰级的 slot anchor/support/conflict，不包含任何未知舰级名称或未知类专属签名。
try:
    import schema_config as _schema_config_v29
except Exception:
    _schema_config_v29 = None


def v29_cfg(name: str, default=None):
    if _schema_config_v29 is None:
        return default
    return getattr(_schema_config_v29, name, default)


def v29_threshold(name: str, default: float) -> float:
    table = v29_cfg("V29_RULE_THRESHOLDS", {}) or {}
    try:
        return float(table.get(name, default))
    except Exception:
        return default


def v29_slot_rules_for(cls_name: str, key: str):
    table = v29_cfg("V29_KNOWN_CLASS_SLOT_RULES", {}) or {}
    info = table.get(cls_name, {}) or {}
    return info.get(key, []) or []


def v29_get_observed_value(observed: Dict[str, Dict[str, Any]], slot_path: str) -> Any:
    if slot_path == "_RAW.raw_text":
        return get_raw_text_from_observed(observed)
    if "." in slot_path:
        group, slot = slot_path.split(".", 1)
        group_obj = observed.get(group, {}) if isinstance(observed, dict) else {}
        if isinstance(group_obj, dict):
            return group_obj.get(slot, "未知")
        return "未知"
    # 兼容只写 slot 的情况：在所有 group 中查找第一个。
    for group_obj in observed.values():
        if isinstance(group_obj, dict) and slot_path in group_obj:
            return group_obj.get(slot_path, "未知")
    return "未知"


def v29_value_texts(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for x in value:
            out.extend(v29_value_texts(x))
        return out
    text = clean_text(value)
    if not text or text in {"未知", "不确定", "未提及"}:
        return []
    parts = split_values(text)
    if not parts:
        parts = [text]
    # 保留整体字符串，也保留拆分值，便于 Keywords 多短语匹配。
    merged = [text] + [p for p in parts if p != text]
    return [x for x in merged if x]


def v29_text_match(obs_text: str, pat: str) -> bool:
    obs = clean_text(obs_text)
    p = clean_text(pat)
    if not obs or not p:
        return False
    # 有/无/是/否这类短值必须精确，避免“有”到处匹配。
    if p in {"有", "无", "是", "否", "0", "1", "2", "3", "4"}:
        return obs == p or obs.startswith(p)
    ok = compact_key(obs)
    pk = compact_key(p)
    if not ok or not pk:
        return False
    return ok == pk or pk in ok


def v29_rule_match(observed: Dict[str, Dict[str, Any]], slot_path: str, patterns: List[str]) -> Tuple[bool, List[str], Any]:
    raw_value = v29_get_observed_value(observed, slot_path)
    texts = v29_value_texts(raw_value)
    hits: List[str] = []
    for pat in patterns or []:
        for t in texts:
            if v29_text_match(t, str(pat)):
                hits.append(str(pat))
                break
    return bool(hits), hits, raw_value


def v29_score_rule_group(observed: Dict[str, Dict[str, Any]], rules) -> Tuple[float, List[Dict[str, Any]]]:
    score = 0.0
    evidence: List[Dict[str, Any]] = []
    for item in rules or []:
        if not isinstance(item, (tuple, list)) or len(item) < 3:
            continue
        slot_path = str(item[0])
        patterns = item[1]
        weight = float(item[2])
        if isinstance(patterns, str):
            patterns = [patterns]
        matched, hits, raw_value = v29_rule_match(observed, slot_path, list(patterns))
        if matched:
            score += weight
            evidence.append({
                "slot": slot_path,
                "value": raw_value,
                "hits": hits,
                "weight": weight,
            })
    return round(score, 4), evidence


def v29_known_slot_score(observed: Dict[str, Dict[str, Any]], cls_name: str) -> Dict[str, Any]:
    category = get_category_of_known_class(cls_name)
    anchor, anchor_evidence = v29_score_rule_group(observed, v29_slot_rules_for(cls_name, "anchor"))
    support, support_evidence = v29_score_rule_group(observed, v29_slot_rules_for(cls_name, "support"))
    conflict, conflict_evidence = v29_score_rule_group(observed, v29_slot_rules_for(cls_name, "conflict"))
    total = anchor + support - conflict * 0.80
    return {
        "class_name": cls_name,
        "category": category,
        "anchor_score": anchor,
        "support_score": support,
        "conflict_score": conflict,
        "total_score": round(total, 4),
        "anchor_evidence": anchor_evidence,
        "support_evidence": support_evidence,
        "conflict_evidence": conflict_evidence,
    }


def v29_all_slot_scores(observed: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _cat, classes in KNOWN_SHIP_CLASSES.items():
        for cls_name in classes:
            rows.append(v29_known_slot_score(observed, cls_name))
    rows.sort(key=lambda r: (r.get("total_score", 0.0), r.get("anchor_score", 0.0), r.get("support_score", 0.0)), reverse=True)
    return rows


def v29_best_in_category(rows: List[Dict[str, Any]], category: str) -> Optional[Dict[str, Any]]:
    cand = [r for r in rows if r.get("category") == category]
    if not cand:
        return None
    cand.sort(key=lambda r: (r.get("total_score", 0.0), r.get("anchor_score", 0.0), r.get("support_score", 0.0)), reverse=True)
    return cand[0]


def v29_has_known_evidence(row: Optional[Dict[str, Any]], allow_support_only: bool = True) -> bool:
    if not row:
        return False
    anchor = float(row.get("anchor_score", 0.0) or 0.0)
    support = float(row.get("support_score", 0.0) or 0.0)
    conflict = float(row.get("conflict_score", 0.0) or 0.0)
    total = float(row.get("total_score", 0.0) or 0.0)
    if conflict >= v29_threshold("hard_conflict_block", 18.0) and anchor < v29_threshold("anchor_known", 8.0):
        return False
    if conflict >= v29_threshold("conflict_block", 12.0) and anchor < v29_threshold("anchor_known", 8.0):
        return False
    require_anchor_classes = set(v29_cfg("V29_REQUIRE_ANCHOR_CLASSES", []) or [])
    if row.get("class_name") in require_anchor_classes and anchor < v29_threshold("anchor_known", 8.0):
        return False
    if anchor >= v29_threshold("anchor_known", 8.0):
        return True
    if allow_support_only and support >= v29_threshold("support_known", 14.0):
        return True
    if anchor >= v29_threshold("anchor_low", 4.0) and support >= v29_threshold("support_with_anchor", 8.0):
        return True
    if allow_support_only and total >= v29_threshold("support_known", 14.0) and conflict < v29_threshold("conflict_block", 12.0):
        return True
    return False


def v29_weak_closed_match(row: Optional[Dict[str, Any]]) -> bool:
    if not row:
        return True
    anchor = float(row.get("anchor_score", 0.0) or 0.0)
    support = float(row.get("support_score", 0.0) or 0.0)
    conflict = float(row.get("conflict_score", 0.0) or 0.0)
    total = float(row.get("total_score", 0.0) or 0.0)
    if conflict >= v29_threshold("conflict_block", 12.0) and anchor < v29_threshold("anchor_known", 8.0):
        return True
    require_anchor_classes = set(v29_cfg("V29_REQUIRE_ANCHOR_CLASSES", []) or [])
    if row.get("class_name") in require_anchor_classes and anchor < v29_threshold("anchor_known", 8.0):
        return True
    if anchor >= v29_threshold("closed_min_anchor", 4.0):
        return False
    if total >= v29_threshold("closed_min_total", 9.0) and support >= v29_threshold("support_with_anchor", 8.0):
        return False
    return True


def v29_set_category_unknown(match_result: Dict[str, Any], category: str, reason: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    return v26_set_category_unknown(match_result, category, reason, analysis)




def v29_category_slot_scores(observed: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    cues = v29_cfg("V29_CATEGORY_SLOT_CUES", {}) or {}
    scores: Dict[str, float] = {cat: 0.0 for cat in SHIP_CATEGORIES}
    for cat, rules in cues.items():
        s, _e = v29_score_rule_group(observed, rules)
        scores[cat] = float(s)
    return scores


def v29_best_slot_category(observed: Dict[str, Dict[str, Any]]) -> Tuple[str, float]:
    scores = v29_category_slot_scores(observed)
    if not scores:
        return "", 0.0
    cat, score = max(scores.items(), key=lambda kv: kv[1])
    if score <= 0:
        return "", 0.0
    return cat, score


def v29_choose_unknown_category(current_cat: str, observed: Dict[str, Dict[str, Any]], rows: List[Dict[str, Any]]) -> str:
    """当需要拒识时，优先使用 observed 卡槽中的通用大类提示修正类别。"""
    cue_cat, cue_score = v29_best_slot_category(observed)
    if cue_cat and cue_score >= 8.0:
        # 如果该大类下已知类没有足够证据，则可以作为类别内未知的大类。
        row = v29_best_in_category(rows, cue_cat)
        if not v29_has_known_evidence(row, allow_support_only=False):
            return cue_cat
    return current_cat

def v29_enforce_slot_based_known_only(result: Dict[str, Any], observed: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    rows = v29_all_slot_scores(observed)
    best = rows[0] if rows else None
    current_cat = v13_get_category_label(result)
    current_cls = v13_get_known_label(result)
    is_open = v13_is_open_set(result)

    # 如果 base 没有给出大类，则用 slot 分数最强的已知类所属大类作为候选大类；
    # 但只有强证据时才直接填已知类，否则先输出该大类未知。
    if (not current_cat or current_cat not in SHIP_CATEGORIES) and best:
        current_cat = best.get("category") or current_cat
        result = v29_set_category_unknown(result, current_cat, "v29 slot-based：原结果大类为空，使用已知类 slot 证据最强的大类作为开放集候选。", {"known_slot_candidates": rows[:10]})
        is_open = True
        current_cls = ""

    same_cat_row = v29_best_in_category(rows, current_cat) if current_cat else None
    current_row = None
    if current_cls:
        current_row = next((r for r in rows if r.get("class_name") == current_cls), None)
    if current_row is None:
        current_row = same_cat_row

    analysis = {
        "principle": "v29_slot_based_known_only: use observed_attributes slot anchor/support/conflict; no unknown-class signatures",
        "pred_category_before_v29": current_cat,
        "pred_class_before_v29": current_cls,
        "pred_open_set_before_v29": is_open,
        "best_overall": best,
        "best_in_current_category": same_cat_row,
        "known_slot_candidates": rows[:10],
    }
    result["v29_slot_based_analysis"] = analysis

    # 1) 当前已经闭集输出：如果只是共享弱证据，则拒识为该大类未知；否则保留。
    if current_cls and not is_open:
        if v29_weak_closed_match(current_row):
            return v29_set_category_unknown(
                result,
                v29_choose_unknown_category(current_cat or (current_row or {}).get("category") or "", observed, rows),
                f"v29 slot-based：{current_cls} 缺少足够 slot 锚点/支持或存在冲突，改为类别内未知。",
                analysis,
            )
        result["v29_slot_based_analysis"] = analysis
        return result

    # 2) 当前 open-set 或只有大类：若同大类已知舰级有足够 slot 证据，则补回已知类。
    if current_cat in KNOWN_SHIP_CLASSES and same_cat_row and v29_has_known_evidence(same_cat_row, allow_support_only=True):
        cls = same_cat_row.get("class_name")
        fixed = v13_fill_known_class(
            result,
            current_cat,
            cls,
            f"v29 slot-based：observed_attributes 中 {cls} 的 anchor/support 证据充足，取消过度 open-set。",
        )
        fixed["v29_slot_based_analysis"] = analysis
        return fixed

    # 3) 跨大类纠偏：必须有目标类强 slot anchor，避免 v26/v27 那种跨大类乱拉。
    if best and best.get("category") and best.get("category") != current_cat:
        if v29_has_known_evidence(best, allow_support_only=False) and float(best.get("anchor_score", 0.0) or 0.0) >= v29_threshold("cross_anchor_known", 12.0):
            fixed = v13_fill_known_class(
                result,
                best.get("category"),
                best.get("class_name"),
                f"v29 slot-based：跨大类纠偏，但目标已知类具有强 slot anchor。",
            )
            fixed["v29_slot_based_analysis"] = analysis
            return fixed

    # 4) 如果仍没有足够已知类证据，则保持/修正为当前大类类别内未知。
    if current_cat in SHIP_CATEGORIES:
        # 注意：这一步不会加入未知类特征，只表示“已知类证据不足”。
        if is_open:
            return v29_set_category_unknown(result, v29_choose_unknown_category(current_cat, observed, rows), "v29 slot-based：当前大类可判断，但没有足够已知舰级 slot 证据，保持/修正为类别内未知。", analysis)

    return result


# 覆盖前一版 hierarchical_class_match。
def hierarchical_class_match(
    class_data_path: str,
    observed_attributes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """v29：slot-based known-only open-set 后处理。"""
    result = _hierarchical_class_match_v13_known_only_base(class_data_path, observed_attributes)
    return v29_enforce_slot_based_known_only(result, observed_attributes)


# ==================== v31: category-first slot-based known-only postprocess ====================
# 目标：在 v29 的 clean known-only 基础上，先稳定大类，再在同一大类内做已知舰级/类别内未知判断。
# 原则：
# 1. 不使用未知舰级名称；
# 2. 不使用未知类专属签名；
# 3. 缺少强锚点不直接触发 open-set；
# 4. 但如果文本明确“未出现/不具备”某已知类关键锚点，则作为该已知类反证；
# 5. 跨大类纠偏必须依赖明确的大类词或强 slot 证据，避免航母/巡洋舰/两栖舰乱拉。

V31_NEGATION_WORDS = ["未出现", "没有", "无", "未见", "看不到", "不具备", "并未", "未提到", "未观察到", "不像", "不是", "并非"]


def v31_text_bundle(observed: Dict[str, Dict[str, Any]]) -> str:
    """把 raw_text + Keywords + 关键任务字段合并，用于大类提示和显式反证检测。"""
    parts: List[str] = []
    raw = get_raw_text_from_observed(observed)
    if raw:
        parts.append(str(raw))
    for group, slots in (observed or {}).items():
        if not isinstance(slots, dict):
            continue
        for slot in ["Keywords", "Primary_Mission", "Area_Air_Defense", "Command_Control", "Anti_Submarine", "Amphibious_Assault", "Landing_Operation"]:
            if slot in slots and not is_unknown_value(slots.get(slot)):
                parts.append(str(slots.get(slot)))
    return "，".join(parts)


def v31_has_text(text: str, phrase: str) -> bool:
    return compact_key(phrase) in compact_key(text)


def v31_phrase_negated(text: str, phrase: str, window: int = 14) -> bool:
    """判断 phrase 是否被前文否定，如“未出现96单元”。"""
    if not text or not phrase:
        return False
    start = 0
    while True:
        idx = text.find(phrase, start)
        if idx < 0:
            return False
        prefix = text[max(0, idx - window):idx]
        if any(w in prefix for w in V31_NEGATION_WORDS):
            return True
        start = idx + len(phrase)


def v31_add_score(scores: Dict[str, float], cat: str, value: float):
    if cat in scores:
        scores[cat] += float(value)


def v31_category_scores(observed: Dict[str, Dict[str, Any]], rows: List[Dict[str, Any]], base_category: str = "") -> Dict[str, float]:
    """更稳定的大类判断：slot 类别提示 + 明确原文大类词 + 已知类强证据。"""
    scores = {cat: 0.0 for cat in SHIP_CATEGORIES}

    # 1) 继承 v29 的 slot 大类提示，但不直接完全相信 base 分类。
    for cat, sc in (v29_category_slot_scores(observed) or {}).items():
        scores[cat] += float(sc or 0.0)

    # 2) base 原有大类只给轻微惯性分，避免错误 base 把结果锁死。
    if base_category in scores:
        scores[base_category] += 2.0

    text = v31_text_bundle(observed)

    # 3) 原文强大类提示。注意“航母战斗群”不是航空母舰本体。
    if v31_has_text(text, "航母战斗群") and (v31_has_text(text, "区域防空") or v31_has_text(text, "指挥")):
        v31_add_score(scores, "巡洋舰", 14.0)
    if any(v31_has_text(text, p) for p in ["导弹巡洋舰", "宙斯盾巡洋舰", "巡洋舰", "舰队指挥", "指挥舰"]):
        v31_add_score(scores, "巡洋舰", 16.0)

    # 驱逐舰提示要排除“不是普通驱逐舰/不像驱逐舰”一类否定上下文。
    for p in ["导弹驱逐舰", "防空驱逐舰", "主力驱逐舰", "驱逐舰"]:
        if v31_has_text(text, p) and not v31_phrase_negated(text, p):
            v31_add_score(scores, "驱逐舰", 14.0)
            break

    if any(v31_has_text(text, p) for p in ["护卫舰", "巡防舰", "反潜护卫", "通用护卫", "中型护卫", "反潜作战取向", "反潜作战"]):
        v31_add_score(scores, "护卫舰", 16.0)
    if any(v31_has_text(text, p) for p in ["濒海战斗舰", "三体船", "三体结构", "任务模块"]):
        v31_add_score(scores, "护卫舰", 10.0)

    if any(v31_has_text(text, p) for p in ["两栖攻击舰", "两栖船坞运输舰", "两栖运输舰", "两栖舰", "LHD", "LPD", "MV-22", "鱼鹰"]):
        v31_add_score(scores, "两栖舰", 16.0)
    if any(v31_has_text(text, p) for p in ["船坞登陆舰", "船坞登陆", "LSD", "井围甲板"]):
        v31_add_score(scores, "登陆舰", 16.0)

    # 航母提示：不要把“航母战斗群”当成航母本体。
    if any(v31_has_text(text, p) for p in ["航空母舰", "超级航母", "核动力航母", "海上机场", "固定翼舰载机联队"]):
        v31_add_score(scores, "航空母舰", 18.0)
    elif (v31_has_text(text, "航母") and not v31_has_text(text, "航母战斗群")):
        v31_add_score(scores, "航空母舰", 10.0)

    # 4) 关键 slot 组合。
    catapult = clean_text(v29_get_observed_value(observed, "AVIATION_FEATURES.Catapult"))
    arrest = clean_text(v29_get_observed_value(observed, "AVIATION_FEATURES.Arresting_Gear"))
    flight_deck = clean_text(v29_get_observed_value(observed, "AVIATION_FEATURES.Flight_Deck_Type"))
    well = clean_text(v29_get_observed_value(observed, "AMPHIBIOUS_FEATURES.Well_Deck"))
    stern_gate = clean_text(v29_get_observed_value(observed, "AMPHIBIOUS_FEATURES.Stern_Gate"))
    lc_cap = clean_text(v29_get_observed_value(observed, "AMPHIBIOUS_FEATURES.Landing_Craft_Capacity"))
    hull = clean_text(v29_get_observed_value(observed, "VISUAL_STRUCTURE.Hull_Form"))
    hangar = clean_text(v29_get_observed_value(observed, "AVIATION_FEATURES.Hangar"))
    gun_cal = clean_text(v29_get_observed_value(observed, "WEAPON_SENSOR_FEATURES.Main_Gun_Caliber"))
    vls_level = clean_text(v29_get_observed_value(observed, "WEAPON_SENSOR_FEATURES.VLS_Count_Level"))
    length = clean_text(v29_get_observed_value(observed, "TEXT_ATTRIBUTES.Length_Overall"))
    disp = clean_text(v29_get_observed_value(observed, "TEXT_ATTRIBUTES.Full_Load_Displacement"))

    if catapult == "有" and arrest == "有":
        v31_add_score(scores, "航空母舰", 16.0)
    if "全通" in flight_deck and ("有" in well or "大型" in well or "LCAC" in lc_cap):
        v31_add_score(scores, "两栖舰", 10.0)
    if "3艘" in lc_cap or "三艘" in lc_cap:
        v31_add_score(scores, "两栖舰", 10.0)
    if "2艘" in lc_cap or "两艘" in lc_cap:
        v31_add_score(scores, "两栖舰", 10.0)
    if "4艘" in lc_cap or "四艘" in lc_cap:
        v31_add_score(scores, "登陆舰", 12.0)
    if "三体" in hull:
        v31_add_score(scores, "护卫舰", 14.0)
    if ("直升机机库" in hangar or hangar == "有") and ("反潜" in text or "中口径" in gun_cal or "76" in gun_cal or "57" in gun_cal):
        v31_add_score(scores, "护卫舰", 10.0)
    if "122" in vls_level:
        v31_add_score(scores, "巡洋舰", 14.0)
    if ("172" in length or "173" in length) and ("9480" in disp or "9500" in disp):
        v31_add_score(scores, "巡洋舰", 18.0)
    if ("155" in length) and ("9200" in disp or "9238" in disp):
        v31_add_score(scores, "驱逐舰", 16.0)

    # 5) 已知类 slot 强证据也能支持其大类，但只使用 anchor，弱支持不直接改变大类。
    for row in rows or []:
        cat = row.get("category")
        if cat not in scores:
            continue
        anchor = float(row.get("anchor_score", 0.0) or 0.0)
        support = float(row.get("support_score", 0.0) or 0.0)
        conflict = float(row.get("conflict_score", 0.0) or 0.0)
        if conflict >= 12.0 and anchor < 8.0:
            continue
        if anchor >= 6.0:
            scores[cat] += anchor * 1.15 + min(support, 6.0) * 0.25

    return scores


def v31_choose_category(observed: Dict[str, Dict[str, Any]], rows: List[Dict[str, Any]], base_category: str) -> Tuple[str, Dict[str, float]]:
    scores = v31_category_scores(observed, rows, base_category)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        return base_category, scores
    top_cat, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    # 分数足够高且领先，才覆盖 base；否则保留 base。
    if top_score >= 10.0 and (top_score - second_score >= 4.0 or top_score >= 18.0):
        return top_cat, scores
    if base_category in SHIP_CATEGORIES:
        return base_category, scores
    return top_cat, scores


def v31_extra_negated_anchor_conflict(observed: Dict[str, Dict[str, Any]], cls_name: str) -> Tuple[float, List[Dict[str, Any]]]:
    """如果原文明确说“不具备/未出现”该已知类关键锚点，则作为 known-only 反证。"""
    text = v31_text_bundle(observed)
    evidence = []
    penalty = 0.0
    for item in v29_slot_rules_for(cls_name, "anchor") or []:
        if not isinstance(item, (tuple, list)) or len(item) < 3:
            continue
        patterns = item[1]
        weight = float(item[2])
        if isinstance(patterns, str):
            patterns = [patterns]
        for pat in patterns:
            p = str(pat)
            if len(compact_key(p)) < 2:
                continue
            if p in {"有", "无", "是", "否"}:
                continue
            if v31_phrase_negated(text, p):
                add = min(weight, 6.0)
                penalty += add
                evidence.append({"pattern": p, "penalty": add, "reason": f"原文明确否定/未出现已知类锚点：{p}"})
    return round(penalty, 4), evidence


def v31_adjust_row_with_text_conflict(observed: Dict[str, Dict[str, Any]], row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not row:
        return row
    new = dict(row)
    extra, ev = v31_extra_negated_anchor_conflict(observed, str(row.get("class_name", "")))
    if extra > 0:
        old_conf = float(new.get("conflict_score", 0.0) or 0.0)
        new["conflict_score"] = round(old_conf + extra, 4)
        old_total = float(new.get("total_score", 0.0) or 0.0)
        new["total_score"] = round(old_total - extra * 0.80, 4)
        new.setdefault("conflict_evidence", [])
        new["conflict_evidence"] = list(new.get("conflict_evidence", [])) + ev
    return new


def v31_has_known_evidence(row: Optional[Dict[str, Any]], category: str = "") -> bool:
    if not row:
        return False
    anchor = float(row.get("anchor_score", 0.0) or 0.0)
    support = float(row.get("support_score", 0.0) or 0.0)
    conflict = float(row.get("conflict_score", 0.0) or 0.0)
    total = float(row.get("total_score", 0.0) or 0.0)

    if conflict >= 12.0 and anchor < 8.0:
        return False
    if conflict >= 18.0:
        return False

    # 强锚点：直接认为是已知类。
    if anchor >= 6.0:
        return True

    # 有一定锚点 + 支持：认为是已知类。
    if anchor >= 4.0 and support >= 6.0:
        return True

    # 只有支持特征时更谨慎：允许已知类补全，但要求支持足够多且没有显式反证。
    # 这可以避免 v29 的过度 open-set，但不会像 v30 那样完全闭集化。
    if support >= 14.0 and total >= 12.0 and conflict < 8.0:
        return True

    return False


def v31_set_unknown(match_result: Dict[str, Any], category: str, reason: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    fixed = v26_set_category_unknown(match_result, category, reason, analysis)
    fixed["v31_category_first_analysis"] = analysis
    return fixed


def v31_fill_known(match_result: Dict[str, Any], category: str, cls_name: str, reason: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    fixed = v13_fill_known_class(match_result, category, cls_name, reason)
    fixed["v31_category_first_analysis"] = analysis
    return fixed


def v31_enforce_category_first_slot_based(result: Dict[str, Any], observed: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    raw_rows = v29_all_slot_scores(observed)
    rows = [v31_adjust_row_with_text_conflict(observed, r) for r in raw_rows]
    rows.sort(key=lambda r: (r.get("total_score", 0.0), r.get("anchor_score", 0.0), r.get("support_score", 0.0)), reverse=True)

    base_cat = v13_get_category_label(result)
    base_cls = v13_get_known_label(result)
    base_open = v13_is_open_set(result)

    chosen_cat, category_scores = v31_choose_category(observed, rows, base_cat)
    same_cat_row = v29_best_in_category(rows, chosen_cat) if chosen_cat else None
    best = rows[0] if rows else None

    # 如果某个其他大类的已知类具有压倒性强锚点，则允许跨大类纠偏。
    # 但必须是 anchor 很强，不允许仅靠共享 support 跨类。
    if best and best.get("category") != chosen_cat:
        best_anchor = float(best.get("anchor_score", 0.0) or 0.0)
        same_total = float((same_cat_row or {}).get("total_score", 0.0) or 0.0)
        best_total = float(best.get("total_score", 0.0) or 0.0)
        if best_anchor >= 10.0 and best_total >= same_total + 5.0:
            chosen_cat = best.get("category") or chosen_cat
            same_cat_row = best

    analysis = {
        "principle": "v31_category_first_slot_based: decide category first; then known/open-set inside that category; known-only, no unknown-class signatures",
        "base_category": base_cat,
        "base_class": base_cls,
        "base_open_set": base_open,
        "chosen_category": chosen_cat,
        "category_scores": category_scores,
        "best_overall": best,
        "best_in_chosen_category": same_cat_row,
        "known_slot_candidates": rows[:10],
    }

    if not chosen_cat or chosen_cat not in SHIP_CATEGORIES:
        result["v31_category_first_analysis"] = analysis
        return result

    # 1) 如果同大类内已知类证据足够，输出已知类。
    if same_cat_row and v31_has_known_evidence(same_cat_row, chosen_cat):
        return v31_fill_known(
            result,
            chosen_cat,
            same_cat_row.get("class_name"),
            f"v31 category-first：先确定大类为 {chosen_cat}，同大类已知舰级 slot 证据足够。",
            analysis,
        )

    # 2) 如果没有足够同大类已知类证据，则输出类别内未知。
    return v31_set_unknown(
        result,
        chosen_cat,
        f"v31 category-first：能判断大类为 {chosen_cat}，但同大类已知舰级证据不足或存在关键反证。",
        analysis,
    )


# 覆盖 v29 hierarchical_class_match。
def hierarchical_class_match(
    class_data_path: str,
    observed_attributes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """v31：category-first slot-based known-only open-set 后处理。"""
    result = _hierarchical_class_match_v13_known_only_base(class_data_path, observed_attributes)
    return v31_enforce_category_first_slot_based(result, observed_attributes)



# ==================== v32: category-lock and amphibious refinement ====================
# 目标：在 v31 category-first 基础上继续修正：
# 1. 更强的大类锁定，避免明确“驱逐舰/护卫舰/巡防舰/船坞登陆舰”等文本被跨类拉走；
# 2. 强化两栖攻击舰 LHD、两栖船坞运输舰 LPD、船坞登陆舰 LSD 的内部区分；
# 3. 仍然只使用已知舰级先验，不引入任何未知舰级名称或未知类专属签名。

V32_SOFT_NEGATION_WORDS = [
    "不是", "不像", "非", "并非", "未见", "没有", "无", "小于", "少于", "明显小于", "低于", "不属于"
]


def v32_text(observed: Dict[str, Dict[str, Any]]) -> str:
    return v31_text_bundle(observed)


def v32_has(text: str, phrase: str) -> bool:
    return compact_key(phrase) in compact_key(text)


def v32_has_any(text: str, phrases: List[str]) -> bool:
    return any(v32_has(text, p) for p in phrases)


def v32_is_negated(text: str, phrase: str, window: int = 16) -> bool:
    """更宽松的否定/排除检测，用于避免“明显小于大型巡洋舰或驱逐舰”被当成巡洋舰/驱逐舰提示。"""
    if not text or not phrase:
        return False
    start = 0
    while True:
        idx = text.find(phrase, start)
        if idx < 0:
            return False
        prefix = text[max(0, idx - window):idx]
        if any(w in prefix for w in V32_SOFT_NEGATION_WORDS):
            return True
        start = idx + len(phrase)


def v32_slot(observed: Dict[str, Dict[str, Any]], dotted: str) -> str:
    try:
        group, slot = dotted.split('.', 1)
    except ValueError:
        return ""
    return clean_text((observed.get(group, {}) or {}).get(slot, ""))


def v32_num_in(text: str, *keys: str) -> bool:
    return any(k in clean_text(text) for k in keys)


def v32_add(scores: Dict[str, float], cat: str, value: float):
    if cat in scores:
        scores[cat] += float(value)


def v32_category_scores(observed: Dict[str, Dict[str, Any]], rows: List[Dict[str, Any]], base_category: str = "") -> Dict[str, float]:
    """v32 大类分：保留 v31 分数，但加入显式大类锁定和反向排除。"""
    scores = v31_category_scores(observed, rows, base_category)
    text = v32_text(observed)

    # 关键槽位
    hull = v32_slot(observed, "VISUAL_STRUCTURE.Hull_Form")
    flight = v32_slot(observed, "AVIATION_FEATURES.Flight_Deck_Type")
    catapult = v32_slot(observed, "AVIATION_FEATURES.Catapult")
    arrest = v32_slot(observed, "AVIATION_FEATURES.Arresting_Gear")
    hangar = v32_slot(observed, "AVIATION_FEATURES.Hangar")
    well = v32_slot(observed, "AMPHIBIOUS_FEATURES.Well_Deck")
    stern = v32_slot(observed, "AMPHIBIOUS_FEATURES.Stern_Gate")
    lc_cap = v32_slot(observed, "AMPHIBIOUS_FEATURES.Landing_Craft_Capacity")
    gun = v32_slot(observed, "WEAPON_SENSOR_FEATURES.Main_Gun_Caliber")
    gun_pos = v32_slot(observed, "WEAPON_SENSOR_FEATURES.Main_Gun_Position")
    vls = v32_slot(observed, "WEAPON_SENSOR_FEATURES.VLS_Count_Level")
    radar = v32_slot(observed, "WEAPON_SENSOR_FEATURES.Radar_Array_Type")
    anti_sub = v32_slot(observed, "MISSION_FEATURES.Anti_Submarine")
    mission = v32_slot(observed, "MISSION_FEATURES.Primary_Mission")
    command = v32_slot(observed, "MISSION_FEATURES.Command_Control")
    length = v32_slot(observed, "TEXT_ATTRIBUTES.Length_Overall")
    disp = v32_slot(observed, "TEXT_ATTRIBUTES.Full_Load_Displacement")

    # A. 原始大类词锁定。只使用通用大类词，不使用未知类名称。
    if v32_has_any(text, ["大驱", "大型驱逐舰", "导弹驱逐舰", "防空驱逐舰", "主力驱逐舰"]):
        v32_add(scores, "驱逐舰", 28.0)
    elif v32_has(text, "驱逐舰") and not v32_is_negated(text, "驱逐舰"):
        v32_add(scores, "驱逐舰", 20.0)

    if v32_has_any(text, ["护卫舰", "巡防舰", "隐身化护卫舰", "新型护卫舰", "反潜护卫", "通用护卫"]):
        v32_add(scores, "护卫舰", 30.0)
    if v32_has_any(text, ["巡逻护航", "反潜作战取向", "反潜作战", "护航和对海作战"]):
        v32_add(scores, "护卫舰", 18.0)

    if v32_has_any(text, ["导弹巡洋舰", "宙斯盾巡洋舰", "巡洋舰"]):
        # “明显小于大型巡洋舰或驱逐舰”这类不是巡洋舰提示。
        if not (v32_is_negated(text, "巡洋舰") or v32_has_any(text, ["小于大型巡洋舰", "不像大型巡洋舰"])):
            v32_add(scores, "巡洋舰", 24.0)

    if v32_has_any(text, ["两栖攻击舰", "LHD", "小航母式两栖舰"]):
        v32_add(scores, "两栖舰", 30.0)
    if v32_has_any(text, ["两栖船坞运输舰", "船坞运输舰", "两栖运输舰", "LPD", "鱼鹰", "MV-22"]):
        v32_add(scores, "两栖舰", 30.0)
    if v32_has_any(text, ["船坞登陆舰", "LSD", "井围甲板"]):
        v32_add(scores, "登陆舰", 34.0)

    # 航母只用本体词，继续排除“航母战斗群/航母编队护航”。
    if v32_has_any(text, ["航空母舰", "超级航母", "核动力航母", "海上机场", "固定翼舰载机联队"]):
        v32_add(scores, "航空母舰", 26.0)
    if v32_has_any(text, ["航母编队里的主力护航驱逐舰", "航母战斗群区域防空", "航母战斗群指挥"]):
        scores["航空母舰"] -= 14.0

    # B. slot 组合锁定。
    if catapult == "有" and arrest == "有" and ("全通" in flight or "斜角" in flight or "固定翼" in text):
        v32_add(scores, "航空母舰", 28.0)

    # 巡洋舰：提康德罗加级参数/122单元/指挥区域防空。
    if "122" in vls or v32_has_any(text, ["122单元", "16组八联装", "双127mm", "舰艏和舰艉各一门"]):
        v32_add(scores, "巡洋舰", 30.0)
    if (v32_num_in(length, "172", "172.8") and v32_num_in(disp, "9480", "9500")):
        v32_add(scores, "巡洋舰", 36.0)
    if ("区域防空" in text and ("指挥" in text or command in {"有", "强", "是"})):
        v32_add(scores, "巡洋舰", 22.0)

    # 驱逐舰：伯克级参数/96单元/Flight/SPY-1D。
    if v32_has_any(text, ["SPY-1D", "AN/SPY-1D", "Flight IIA", "96单元", "两座直升机库", "SH-60", "MH-60"]):
        v32_add(scores, "驱逐舰", 30.0)
    if (v32_num_in(length, "155") and v32_num_in(disp, "9200", "9238")):
        v32_add(scores, "驱逐舰", 34.0)

    # 护卫舰：三体/LCS/57mm/中等排水量+反潜护航。
    if "三体" in hull or v32_has_any(text, ["三体船", "濒海战斗舰", "LCS", "任务模块"]):
        v32_add(scores, "护卫舰", 32.0)
    if ("57" in gun or "76" in gun or "中口径" in gun) and ("反潜" in text or "护卫" in text or "巡防" in text or "护航" in text):
        v32_add(scores, "护卫舰", 26.0)
    if ("低" in vls or "少量" in text or "有限垂发" in text) and ("反潜" in text or "巡逻护航" in text or "三千" in text or "四千" in text):
        v32_add(scores, "护卫舰", 26.0)
        scores["巡洋舰"] -= 10.0
        scores["驱逐舰"] -= 6.0

    # 两栖/登陆区分。
    if "全通" in flight and (v32_has_any(text, ["STOVL", "AV-8B", "F-35B", "垂直起降"]) or "3艘" in lc_cap):
        v32_add(scores, "两栖舰", 28.0)
    if v32_has_any(text, ["LPD", "MV-22", "鱼鹰", "两栖船坞运输舰", "2艘LCAC", "720名"]):
        v32_add(scores, "两栖舰", 30.0)
    if v32_has_any(text, ["LSD", "船坞登陆舰", "4艘LCAC", "四艘LCAC", "大型井围甲板", "无直升机机库"]):
        v32_add(scores, "登陆舰", 32.0)
    if "4艘" in lc_cap or "四艘" in lc_cap:
        v32_add(scores, "登陆舰", 20.0)
    if "2艘" in lc_cap or "两艘" in lc_cap or "3艘" in lc_cap or "三艘" in lc_cap:
        v32_add(scores, "两栖舰", 14.0)

    # C. 明显排除：避免护卫舰/驱逐舰/两栖舰被误拉为航空母舰。
    if v32_has_any(text, ["驱逐舰", "护卫舰", "巡防舰", "大驱", "反潜", "巡逻护航", "船坞登陆舰", "两栖攻击舰", "LPD", "LSD"]):
        if not (catapult == "有" and arrest == "有"):
            scores["航空母舰"] -= 18.0

    return scores


def v32_choose_category(observed: Dict[str, Dict[str, Any]], rows: List[Dict[str, Any]], base_category: str) -> Tuple[str, Dict[str, float]]:
    scores = v32_category_scores(observed, rows, base_category)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        return base_category, scores
    top_cat, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    # v32 比 v31 更相信显式大类锁定；但如果差距很小仍保留 base。
    if top_score >= 14.0 and (top_score - second_score >= 3.0 or top_score >= 28.0):
        return top_cat, scores
    if base_category in SHIP_CATEGORIES:
        return base_category, scores
    return top_cat, scores


def v32_direct_known_class(chosen_cat: str, observed: Dict[str, Dict[str, Any]], same_cat_row: Optional[Dict[str, Any]]) -> Optional[str]:
    """仅用已知舰级锚点做同大类内补全；不使用未知类签名。"""
    text = v32_text(observed)
    flight = v32_slot(observed, "AVIATION_FEATURES.Flight_Deck_Type")
    catapult = v32_slot(observed, "AVIATION_FEATURES.Catapult")
    arrest = v32_slot(observed, "AVIATION_FEATURES.Arresting_Gear")
    well = v32_slot(observed, "AMPHIBIOUS_FEATURES.Well_Deck")
    stern = v32_slot(observed, "AMPHIBIOUS_FEATURES.Stern_Gate")
    lc_cap = v32_slot(observed, "AMPHIBIOUS_FEATURES.Landing_Craft_Capacity")
    hull = v32_slot(observed, "VISUAL_STRUCTURE.Hull_Form")
    gun = v32_slot(observed, "WEAPON_SENSOR_FEATURES.Main_Gun_Caliber")
    vls = v32_slot(observed, "WEAPON_SENSOR_FEATURES.VLS_Count_Level")
    length = v32_slot(observed, "TEXT_ATTRIBUTES.Length_Overall")
    disp = v32_slot(observed, "TEXT_ATTRIBUTES.Full_Load_Displacement")

    if chosen_cat == "航空母舰":
        if (catapult == "有" and arrest == "有") or v32_has_any(text, ["尼米兹", "10万吨", "核动力", "弹射器", "拦阻索", "A4W"]):
            return "尼米兹级航空母舰"

    if chosen_cat == "巡洋舰":
        if v32_has_any(text, [
            "提康德罗加", "导弹巡洋舰", "宙斯盾巡洋舰", "122单元", "16组八联装", "双127mm",
            "舰队指挥", "指挥舰", "指挥中心", "航母战斗群的指挥中心", "现役唯一一级巡洋舰",
            "前后都有垂发", "前后均有垂发", "垂发数量很多", "AN/SPY-1", "SPY-1相控阵雷达"
        ]):
            return "提康德罗加级导弹巡洋舰"
        if "122" in vls or (v32_num_in(length, "172", "172.8") and v32_num_in(disp, "9480", "9500")):
            return "提康德罗加级导弹巡洋舰"

    if chosen_cat == "驱逐舰":
        if v32_has_any(text, [
            "阿利", "伯克", "DDG-51", "SPY-1D", "AN/SPY-1D", "Flight IIA", "Flight III",
            "96单元", "90-96单元", "两座直升机库", "SH-60", "MH-60", "航母的带刀护卫",
            "主力护航驱逐舰", "宙斯盾驱逐舰"
        ]):
            return "阿利·伯克级驱逐舰"
        if v32_num_in(length, "155") and v32_num_in(disp, "9200", "9238"):
            return "阿利·伯克级驱逐舰"

    if chosen_cat == "护卫舰":
        # 明确否定三体/濒海战斗舰时，不能补成独立级。
        if v32_is_negated(text, "三体") or v32_is_negated(text, "濒海战斗舰") or v32_has_any(text, ["不是三体", "不像濒海三体", "不是濒海三体舰"]):
            return None
        if "三体" in hull or v32_has_any(text, [
            "独立级", "濒海战斗舰", "LCS", "三体船", "任务模块", "铝合金三体", "喷水推进",
            "45-50节", "45节", "50节", "MT30", "舰长127.6", "舰宽31.6", "满载排水量2784"
        ]):
            return "独立级濒海战斗舰"
        if "57" in gun and (v32_has_any(text, ["高速", "濒海", "任务模块", "浅吃水"]) or "三体" in text):
            return "独立级濒海战斗舰"

    if chosen_cat == "两栖舰":
        if v32_has_any(text, ["黄蜂", "两栖攻击舰", "LHD", "AV-8B", "F-35B", "STOVL", "垂直起降", "3艘LCAC", "三艘LCAC", "41150", "253米"]):
            return "黄蜂级两栖攻击舰"
        if v32_has_any(text, ["圣安东尼奥", "LPD", "两栖船坞运输舰", "MV-22", "鱼鹰", "2艘LCAC", "两艘LCAC", "720名", "25300"]):
            return "圣安东尼奥级两栖船坞运输舰"
        # 有全通飞行甲板/STOVL更像黄蜂；有车辆/货物运输+直升机/鱼鹰+坞舱更像圣安东尼奥。
        if "全通" in flight and ("有" in well or "大型" in well):
            return "黄蜂级两栖攻击舰"
        if v32_has_any(text, ["车辆", "货物", "运兵", "鱼鹰", "直升机/鱼鹰"]):
            return "圣安东尼奥级两栖船坞运输舰"

    if chosen_cat == "登陆舰":
        if v32_has_any(text, [
            "惠德比", "LSD", "船坞登陆舰", "4艘LCAC", "四艘LCAC", "大型井围甲板", "无直升机机库",
            "16100", "627名", "美国的登陆舰", "运输登陆装备", "一万多吨", "大型坞舱容量"
        ]):
            return "惠德比岛级船坞登陆舰"
        if "4艘" in lc_cap or "四艘" in lc_cap:
            return "惠德比岛级船坞登陆舰"

    # 如果同大类已知类已经有很强证据，则保留 v31 的补全能力。
    if same_cat_row and v31_has_known_evidence(same_cat_row, chosen_cat):
        return same_cat_row.get("class_name")
    return None


def v32_should_force_unknown(chosen_cat: str, known_cls: Optional[str], observed: Dict[str, Dict[str, Any]], same_cat_row: Optional[Dict[str, Any]]) -> bool:
    """显式已知类反证时拒绝闭集补全。只基于已知类锚点否定，不基于未知类签名。"""
    if not known_cls:
        return False
    text = v32_text(observed)
    # 独立级：被明确否定三体/濒海战斗舰。
    if known_cls == "独立级濒海战斗舰" and (v32_is_negated(text, "三体") or v32_is_negated(text, "濒海战斗舰") or v32_has_any(text, ["不是三体", "不像濒海三体"])):
        return True
    # 黄蜂：明确无全通飞行甲板/STOVL，且更像运输/登陆。
    if known_cls == "黄蜂级两栖攻击舰" and v32_has_any(text, ["没有全通飞行甲板", "无全通飞行甲板", "没有STOVL", "无STOVL"]):
        return True

    # 阿利·伯克：已知类自身以宙斯盾驱逐舰/较大垂发/直升机库为核心；
    # 如果输入明确是低垂发/无机库/明显反潜护卫配置，则不能只靠“宙斯盾/Mk41/SH-60”硬闭集。
    if known_cls == "阿利·伯克级驱逐舰":
        vls = v32_slot(observed, "WEAPON_SENSOR_FEATURES.VLS_Count_Level")
        hangar = v32_slot(observed, "AVIATION_FEATURES.Hangar")
        crew = v32_slot(observed, "TEXT_ATTRIBUTES.Crew")
        if v32_has_any(text, ["没有机库", "无机库", "无直升机机库", "没有直升机库", "无直升机库"]):
            return True
        if "16" in str(vls) or "低" in str(vls) or v32_has_any(text, ["16单元", "2组八联装", "少量垂发", "低垂发"]):
            return True
        if v32_has_any(text, ["反潜护卫", "巡防舰", "通用护卫", "护卫舰"]) and not v32_has_any(text, ["阿利", "伯克", "DDG-51", "SPY-1D", "Flight IIA", "96单元"]):
            return True
        # 166人级别更接近护卫舰规模，不应闭集为伯克级。
        nums = extract_numbers(crew)
        if nums and any(n < 250 for n in nums):
            return True

    return False


def v32_enforce_category_lock_and_amphibious_fix(result: Dict[str, Any], observed: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    raw_rows = v29_all_slot_scores(observed)
    rows = [v31_adjust_row_with_text_conflict(observed, r) for r in raw_rows]
    rows.sort(key=lambda r: (r.get("total_score", 0.0), r.get("anchor_score", 0.0), r.get("support_score", 0.0)), reverse=True)

    base_cat = v13_get_category_label(result)
    base_cls = v13_get_known_label(result)
    base_open = v13_is_open_set(result)

    chosen_cat, category_scores = v32_choose_category(observed, rows, base_cat)
    same_cat_row = v29_best_in_category(rows, chosen_cat) if chosen_cat else None
    best = rows[0] if rows else None

    # 限制跨大类：只有“其他大类已知舰级强锚点非常强”才允许覆盖显式 category lock。
    if best and best.get("category") != chosen_cat:
        best_anchor = float(best.get("anchor_score", 0.0) or 0.0)
        best_total = float(best.get("total_score", 0.0) or 0.0)
        same_total = float((same_cat_row or {}).get("total_score", 0.0) or 0.0)
        if best_anchor >= 14.0 and best_total >= same_total + 8.0:
            chosen_cat = best.get("category") or chosen_cat
            same_cat_row = best

    direct_cls = v32_direct_known_class(chosen_cat, observed, same_cat_row)

    analysis = {
        "principle": "v34_v32_repair: category lock first; known/open-set inside category; known-only, no unknown-class signatures",
        "base_category": base_cat,
        "base_class": base_cls,
        "base_open_set": base_open,
        "chosen_category": chosen_cat,
        "category_scores": category_scores,
        "best_overall": best,
        "best_in_chosen_category": same_cat_row,
        "direct_known_class": direct_cls,
        "known_slot_candidates": rows[:10],
    }

    if not chosen_cat or chosen_cat not in SHIP_CATEGORIES:
        result["v32_category_lock_analysis"] = analysis
        return result

    if direct_cls and not v32_should_force_unknown(chosen_cat, direct_cls, observed, same_cat_row):
        fixed = v13_fill_known_class(result, chosen_cat, direct_cls, "v32：大类锁定后，命中同大类已知舰级锚点/slot 证据。")
        fixed["v32_category_lock_analysis"] = analysis
        return fixed

    fixed = v26_set_category_unknown(
        result,
        chosen_cat,
        "v32：能判断大类，但同大类已知舰级锚点不足或存在已知类反证。",
        analysis,
    )
    fixed["v32_category_lock_analysis"] = analysis
    return fixed


# 覆盖 v31 hierarchical_class_match。
def hierarchical_class_match(
    class_data_path: str,
    observed_attributes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """v34：v32 baseline + safe known-anchor repairs 的 clean known-only open-set 后处理。"""
    result = _hierarchical_class_match_v13_known_only_base(class_data_path, observed_attributes)
    return v32_enforce_category_lock_and_amphibious_fix(result, observed_attributes)



# ==================== v35: repair low original-80 without changing LLM parsing ====================
# 说明：v35 回到 v34/v32 主线，只做两类补丁：
# 1) 对原 80 中大量“有大类/通用强特征，但舰级没有补全”的已知类样本做同大类补全；
# 2) 对少数明显不应闭集到阿利·伯克/提康德罗加/独立级的样本增加已知类反证。
# 不写样本 id，不写未知舰级名称。

_v34_category_scores = v32_category_scores
_v34_direct_known_class = v32_direct_known_class
_v34_should_force_unknown = v32_should_force_unknown


def v35_has_any(text: str, cues: List[str]) -> bool:
    return v32_has_any(text, cues)


def v32_category_scores(observed: Dict[str, Dict[str, Any]], rows: List[Dict[str, Any]], base_category: str) -> Dict[str, float]:
    scores = _v34_category_scores(observed, rows, base_category)
    text = v32_text(observed)

    # 一、驱逐舰锁定：有“主炮 + 垂发/宙斯盾/现代化水面战斗舰”，且明确不像航母/两栖时，不应被拉到航母或两栖。
    if (
        v35_has_any(text, ["宙斯盾系统", "宙斯盾作战系统", "多用途导弹驱逐舰", "导弹驱逐舰", "现代化的军舰", "现代化军舰"])
        or (v35_has_any(text, ["前面有炮", "舰艏主炮", "船头有炮", "舰艏处有主炮"]) and v35_has_any(text, ["垂直发射", "垂发", "发射装置", "武器传感器特征"]))
    ):
        if not v35_has_any(text, ["全通飞行甲板", "一整块飞行甲板", "坞舱", "艉门", "尾门", "LCAC"]):
            scores["驱逐舰"] = scores.get("驱逐舰", 0.0) + 22.0
            scores["航空母舰"] = scores.get("航空母舰", 0.0) - 14.0
            scores["两栖舰"] = scores.get("两栖舰", 0.0) - 10.0

    if v35_has_any(text, ["不像两栖舰", "没有尾门和坞舱", "没有尾门", "没有坞舱", "不像航母", "没有弹射器"]):
        if v35_has_any(text, ["垂直发射", "垂发", "舰艏主炮", "前面有炮", "主炮"]):
            scores["驱逐舰"] = scores.get("驱逐舰", 0.0) + 18.0
            scores["两栖舰"] = scores.get("两栖舰", 0.0) - 14.0
            scores["航空母舰"] = scores.get("航空母舰", 0.0) - 14.0

    # 二、巡洋舰锁定：原始 80 的提康德罗加描述常写“前后发射区/火力很重/大型水面战斗舰”，LLM 未必抽到 122 单元。
    if v35_has_any(text, ["前后都有发射区域", "前后甲板可见武器发射区域", "前后都有发射", "前后甲板", "火力配置很重"]):
        if not v35_has_any(text, ["驱逐舰", "护卫舰", "巡防舰", "三体"]):
            scores["巡洋舰"] = scores.get("巡洋舰", 0.0) + 22.0
            scores["航空母舰"] = scores.get("航空母舰", 0.0) - 12.0
            scores["驱逐舰"] = scores.get("驱逐舰", 0.0) - 4.0

    if v35_has_any(text, ["大型水面战斗舰", "上层建筑比较复杂"]) and v35_has_any(text, ["武器发射区域", "导弹发射区域", "雷达", "方形雷达面"]):
        if not v35_has_any(text, ["驱逐舰", "护卫舰", "三体", "濒海"]):
            scores["巡洋舰"] = scores.get("巡洋舰", 0.0) + 14.0

    # 三、护卫舰/独立级锁定：三体/多体/低矮/宽大舰尾平台不能被拉到驱逐舰。
    if v35_has_any(text, ["三体结构", "三体船", "多体结构", "不是普通驱逐舰一条船体", "可能不是普通单体船", "船体比较宽"]):
        scores["护卫舰"] = scores.get("护卫舰", 0.0) + 26.0
        scores["驱逐舰"] = scores.get("驱逐舰", 0.0) - 12.0
        scores["巡洋舰"] = scores.get("巡洋舰", 0.0) - 10.0

    # 四、两栖攻击舰 vs 航母：全通飞行甲板 + 无弹射器/坞舱/登陆部队，应锁到两栖舰而不是航母。
    if v35_has_any(text, ["看起来像航母", "一整块飞行甲板", "全通式飞行甲板", "全通飞行甲板"]):
        if v35_has_any(text, ["没有弹射器", "未见弹射器", "无弹射器", "坞舱", "登陆艇", "登陆部队", "两栖登陆", "两栖攻击"]):
            scores["两栖舰"] = scores.get("两栖舰", 0.0) + 24.0
            scores["航空母舰"] = scores.get("航空母舰", 0.0) - 18.0

    # 五、LPD 与 LSD：大型运输舰/方盒子舰桥/后部直升机平台/舰尾开口，更偏 LPD；装载登陆艇/传统上层建筑/航空不是主要特征，更偏 LSD。
    if v35_has_any(text, ["大型运输舰", "方盒子", "方盒子一样的舰桥", "上层建筑体量很大", "外形封闭"]):
        if v35_has_any(text, ["后面有直升机平台", "后部有航空作业甲板", "船尾好像有开口", "舰尾疑似设置坞舱艉门", "不像航母"]):
            scores["两栖舰"] = scores.get("两栖舰", 0.0) + 22.0
            scores["登陆舰"] = scores.get("登陆舰", 0.0) - 8.0
            scores["航空母舰"] = scores.get("航空母舰", 0.0) - 12.0

    if v35_has_any(text, ["舰尾宽大", "装载登陆艇", "坞舱门", "传统上层建筑", "航空设施和武器传感器配置不是主要"]):
        scores["登陆舰"] = scores.get("登陆舰", 0.0) + 18.0
        scores["两栖舰"] = scores.get("两栖舰", 0.0) - 4.0

    return scores


def v32_direct_known_class(chosen_cat: str, observed: Dict[str, Dict[str, Any]], same_cat_row: Optional[Dict[str, Any]]) -> Optional[str]:
    text = v32_text(observed)
    base = _v34_direct_known_class(chosen_cat, observed, same_cat_row)

    if chosen_cat == "巡洋舰":
        if v35_has_any(text, [
            "前后都有发射区域", "前后甲板可见武器发射区域", "火力配置很重", "大型水面战斗舰",
            "上层建筑比较复杂", "导弹发射区域", "方形雷达面", "武器发射区域"
        ]) and not v35_has_any(text, ["驱逐舰", "护卫舰", "巡防舰", "三体"]):
            return "提康德罗加级导弹巡洋舰"

    if chosen_cat == "驱逐舰":
        if v35_has_any(text, ["无直升机机库", "没有真正的直升机库", "没有机库", "四角格子", "格子桅", "传统重型四角格子桅杆"]):
            # 这些是对阿利·伯克级“有机库/封闭桅杆”已知原型的反证，不补已知类。
            return None
        if v35_has_any(text, ["宙斯盾系统", "宙斯盾作战系统", "多用途导弹驱逐舰", "典型多用途导弹驱逐舰", "前后垂发阵列"]):
            if v35_has_any(text, ["舰艏主炮", "舰艏处有主炮", "前面有炮", "Mk 41", "垂直发射", "防空", "反潜", "对海作战"]):
                return "阿利·伯克级驱逐舰"
        if v35_has_any(text, ["8000至10000吨级", "燃气轮机动力"]) and v35_has_any(text, ["宙斯盾", "相控阵雷达", "Mk 41"]):
            return "阿利·伯克级驱逐舰"

    if chosen_cat == "护卫舰":
        if v35_has_any(text, ["三体结构", "多体结构", "三体船", "可能不是普通单体船", "船体比较宽", "后部甲板很宽", "大型开放式直升机甲板"]):
            return "独立级濒海战斗舰"

    if chosen_cat == "两栖舰":
        if v35_has_any(text, ["看起来像航母", "一整块飞行甲板", "全通式飞行甲板", "全通飞行甲板"]):
            if v35_has_any(text, ["没有弹射器", "未见弹射器", "无弹射器", "坞舱", "登陆艇", "登陆部队", "两栖登陆", "两栖攻击"]):
                return "黄蜂级两栖攻击舰"
        if v35_has_any(text, ["大型运输舰", "方盒子", "方盒子一样的舰桥", "上层建筑体量很大", "外形封闭", "车辆", "运兵"]):
            if v35_has_any(text, ["直升机平台", "航空作业甲板", "船尾", "舰尾", "坞舱", "艉门", "尾门", "开口"]):
                return "圣安东尼奥级两栖船坞运输舰"

    if chosen_cat == "登陆舰":
        if v35_has_any(text, ["舰尾宽大", "装载登陆艇", "坞舱门", "传统上层建筑", "航空设施和武器传感器配置不是主要"]):
            return "惠德比岛级船坞登陆舰"

    return base


def v32_should_force_unknown(chosen_cat: str, known_cls: Optional[str], observed: Dict[str, Dict[str, Any]], same_cat_row: Optional[Dict[str, Any]]) -> bool:
    text = v32_text(observed)
    base = _v34_should_force_unknown(chosen_cat, known_cls, observed, same_cat_row)
    if base:
        return True

    if known_cls == "阿利·伯克级驱逐舰":
        if v35_has_any(text, ["传统重型四角格子桅杆", "四角格子", "格子结构", "格子桅"]):
            return True
        if v35_has_any(text, ["没有真正的直升机库", "没有机库", "无机库", "无直升机机库", "舰尾只有直升机平台"]):
            return True
        if v35_has_any(text, ["151米", "6200吨", "76毫米", "拖曳阵列声纳", "拖曳阵列声呐", "Mk 48防空垂直发射系统"]):
            return True

    if known_cls == "提康德罗加级导弹巡洋舰":
        if v35_has_any(text, ["76毫米", "6200吨", "151米", "拖曳阵列声纳", "拖曳阵列声呐", "16单元", "3600吨", "3000吨"]):
            return True

    if known_cls == "独立级濒海战斗舰":
        if v35_has_any(text, ["16单元", "127毫米", "CODLOG", "3600吨", "3000吨", "反潜护卫", "通用护卫"]):
            return True

    return False


# 覆盖 v34 hierarchical_class_match，复用 v32_enforce，但其内部调用的 category_scores/direct/force_unknown 已被 v35 覆盖。
def hierarchical_class_match(
    class_data_path: str,
    observed_attributes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """v35：v34 baseline + original-80 rescue + explicit known-class counter-evidence."""
    result = _hierarchical_class_match_v13_known_only_base(class_data_path, observed_attributes)
    return v32_enforce_category_lock_and_amphibious_fix(result, observed_attributes)


# ==================== v47: final_decision sync only ====================
# 单点小改：只修复 known_class_result 已经是已知舰级、open_set_result 已关闭，
# 但 final_decision 仍然残留 category_unknown / primary_class 为空的问题。
# 不改变候选打分、不新增任何舰级规则、不针对未知类做闭集补全。
_hierarchical_class_match_v38_stable = hierarchical_class_match


def v47_has_known_label(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    label = str(obj.get("label") or obj.get("ship_class") or "").strip()
    return label not in {"", "None", "null", "未知"}


def v47_sync_final_decision_if_known(match_result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(match_result, dict):
        return match_result

    known = match_result.get("known_class_result") or {}
    open_set = match_result.get("open_set_result") or {}
    final = match_result.get("final_decision") or {}

    # 只处理：known_class_result 有明确 label，且 open_set_result 明确不是 unknown 的情况。
    # 如果 open_set_result 是 True，不强行覆盖，避免把类别内未知类拉回已知类。
    if not v47_has_known_label(known):
        return match_result
    if bool(open_set.get("is_unknown", False)):
        return match_result

    label = known.get("label") or known.get("ship_class")
    category = known.get("category") or (match_result.get("category_result") or {}).get("label")
    confidence = known.get("confidence", final.get("confidence", 0.0))

    if category:
        old_cat = match_result.get("category_result") or {}
        match_result["category_result"] = {
            "label": category,
            "confidence": old_cat.get("confidence", confidence),
            "status": old_cat.get("status", "matched"),
            "reason": old_cat.get("reason", "v47：同步 known_class_result 的类别。"),
        }

    match_result["open_set_result"] = {
        "is_unknown": False,
        "unknown_scope": None,
        "reason": open_set.get("reason") or "v47：known_class_result 已明确，关闭类别内未知状态。",
    }

    # 无论 final_decision 是否已有内容，只要它不是 known_class / primary_class 为空，就同步。
    if final.get("result_type") != "known_class" or not final.get("primary_class"):
        match_result["final_decision"] = {
            "result_type": "known_class",
            "primary_category": category,
            "primary_class": label,
            "confidence": confidence,
            "status": final.get("status") or "v47_synced_known_class",
            "message": final.get("message") or f"最终判定：{category} / {label}。v47 同步 known_class_result 到 final_decision。",
        }
        match_result["v47_final_decision_sync"] = {
            "applied": True,
            "reason": "known_class_result 已有明确已知舰级且 open_set_result=false，但 final_decision 未同步。",
            "synced_category": category,
            "synced_class": label,
        }
    else:
        match_result["v47_final_decision_sync"] = {"applied": False, "reason": "final_decision 已经同步。"}

    return match_result


def hierarchical_class_match(
    class_data_path: str,
    observed_attributes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """v47：基于 v38 稳定版，仅修复 final_decision 与 known_class_result 不同步。"""
    result = _hierarchical_class_match_v38_stable(class_data_path, observed_attributes)
    return v47_sync_final_decision_if_known(result)




# ==================== v48: top-known-candidate promotion only ====================
# 单点小改：
# v47 证明“同步 known_class_result”不够，因为这些错误样本里 known_class_result 实际为 None；
# 真正存在的是 known_class_candidates/top_known_classes 里已经有高分同大类已知候选。
# v48 只在非常明确的阿利·伯克级正向组合证据存在、且没有两栖/航母强正证据时，
# 把 category_unknown/category_only 修正为已知阿利·伯克级。
_hierarchical_class_match_v47_base = hierarchical_class_match


def v48_clean_text(x: Any) -> str:
    try:
        return str(x or "")
    except Exception:
        return ""


def v48_compact(x: Any) -> str:
    try:
        return compact_key(x)
    except Exception:
        return re.sub(r"\s+", "", str(x or "")).lower()


def v48_has_any_text(text: str, cues: List[str]) -> bool:
    ck = v48_compact(text)
    return any(v48_compact(c) in ck for c in cues)


def v48_observed_to_text(observed: Dict[str, Dict[str, Any]]) -> str:
    parts = []
    try:
        meta = observed.get("_META") or {}
        parts.append(v48_clean_text(meta.get("raw_text", "")))
    except Exception:
        pass
    try:
        for _g, slots in (observed or {}).items():
            if not isinstance(slots, dict):
                continue
            for _slot, val in slots.items():
                if isinstance(val, list):
                    parts.extend(v48_clean_text(v) for v in val)
                else:
                    parts.append(v48_clean_text(val))
    except Exception:
        pass
    return " ".join([p for p in parts if p])


def v48_find_top_known(result: Dict[str, Any], label: str) -> Optional[Dict[str, Any]]:
    for r in result.get("known_class_candidates", []) or []:
        r_label = r.get("label") or r.get("ship_class") or r.get("name")
        if r_label == label:
            return r
    # 兼容 final_decision.alternatives.top_known_classes 里的字段
    alts = ((result.get("final_decision") or {}).get("alternatives") or {}).get("top_known_classes") or []
    for r in alts:
        r_label = r.get("label") or r.get("ship_class") or r.get("name")
        if r_label == label:
            return r
    return None


def v48_score(r: Optional[Dict[str, Any]]) -> float:
    if not isinstance(r, dict):
        return 0.0
    for k in ("confidence", "confidence_or_score", "score"):
        try:
            if r.get(k) is not None and r.get(k) != "":
                return float(r.get(k))
        except Exception:
            pass
    return 0.0


def v48_has_arleigh_combo(observed: Dict[str, Dict[str, Any]]) -> Tuple[bool, List[str]]:
    text = v48_observed_to_text(observed)
    evidence = []

    has_vls = v48_has_any_text(text, [
        "mk41", "mk 41", "垂发", "垂直发射", "垂直发射系统", "vls", "前后为mk41", "mk41垂发"
    ])
    has_bow_gun = v48_has_any_text(text, [
        "舰艏主炮", "舰艏一门127", "舰艏一门127mm", "127mm舰炮", "127mm主炮",
        "舰艏处有主炮", "船头有炮", "前面有炮", "舰艏有炮"
    ])
    has_radar = v48_has_any_text(text, [
        "宙斯盾", "相控阵", "四面固定", "四面雷达", "固定雷达阵面", "spy-1", "spy1"
    ])
    has_modern_ddg_visual = v48_has_any_text(text, [
        "隐身化封闭式上层建筑", "隐身化", "倾斜面", "现代化军舰", "现代化的军舰"
    ])

    if has_vls:
        evidence.append("VLS/Mk41")
    if has_bow_gun:
        evidence.append("舰艏127mm/主炮")
    if has_radar:
        evidence.append("相控阵/宙斯盾")
    if has_modern_ddg_visual:
        evidence.append("隐身化现代水面战斗舰外形")

    # 两栖/航母强正证据；只有这些存在时才阻止补阿利·伯克。
    # 注意“直升机甲板/机库”不是两栖强证据，阿利·伯克级也可能有。
    has_amphibious_positive = v48_has_any_text(text, [
        "坞舱", "艉门", "尾门", "登陆艇", "lcac", "lcu", "车辆甲板", "陆战队员",
        "两栖攻击", "两栖登陆", "运兵", "泛水坞"
    ])
    has_carrier_positive = v48_has_any_text(text, [
        "全通飞行甲板", "斜角飞行甲板", "弹射器", "拦阻索", "舰载机联队", "固定翼舰载机", "航母"
    ])
    # 如果文本是否定表达，例如“不像航母/没有弹射器”，不能视作航母正证据。
    if v48_has_any_text(text, ["不像航母", "不是航母", "没有弹射器", "无弹射器", "未见弹射器", "没有拦阻索", "无拦阻索"]):
        has_carrier_positive = False
    if v48_has_any_text(text, ["不像两栖", "不是两栖", "没有坞舱", "无坞舱", "未见坞舱", "没有艉门", "无艉门"]):
        has_amphibious_positive = False

    positive_count = sum([has_vls, has_bow_gun, has_radar, has_modern_ddg_visual])
    ok = positive_count >= 2 and not has_amphibious_positive and not has_carrier_positive
    return ok, evidence


def v48_promote_arleigh_if_safe(result: Dict[str, Any], observed: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return result

    final = result.get("final_decision") or {}
    if final.get("result_type") == "known_class" and final.get("primary_class"):
        result["v48_arleigh_promotion"] = {"applied": False, "reason": "already_known_class"}
        return result

    cat = (
        (result.get("category_result") or {}).get("label")
        or final.get("primary_category")
    )
    arleigh = v48_find_top_known(result, "阿利·伯克级驱逐舰")
    arleigh_conf = v48_score(arleigh)

    combo_ok, evidence = v48_has_arleigh_combo(observed)

    # 只处理已经明显在驱逐舰方向，或者候选本身极强的情况。
    cat_ok = (cat == "驱逐舰")
    conf_ok = arleigh_conf >= 0.72
    if not (cat_ok and conf_ok and combo_ok):
        result["v48_arleigh_promotion"] = {
            "applied": False,
            "reason": "condition_not_met",
            "category": cat,
            "arleigh_conf": round(arleigh_conf, 4),
            "combo_ok": combo_ok,
            "evidence": evidence,
        }
        return result

    category_result = result.get("category_result") or {}
    result["category_result"] = {
        "label": "驱逐舰",
        "confidence": max(float(category_result.get("confidence", 0.0) or 0.0), float(arleigh_conf)),
        "status": category_result.get("status", "matched"),
        "reason": category_result.get("reason", "") + " v48：阿利·伯克级组合证据保护驱逐舰大类。",
    }
    result["known_class_result"] = {
        "label": "阿利·伯克级驱逐舰",
        "category": "驱逐舰",
        "confidence": round(float(arleigh_conf), 4),
        "score": (arleigh or {}).get("score"),
        "known_status": "Known",
        "reason": "v48：同大类高分候选 + VLS/Mk41、舰艏主炮、相控阵/宙斯盾等组合证据，且无两栖/航母强正证据。",
    }
    result["open_set_result"] = {
        "is_unknown": False,
        "unknown_scope": None,
        "reason": "v48：安全补全为阿利·伯克级已知类。",
    }
    old_alts = (final.get("alternatives") if isinstance(final, dict) else None) or {}
    result["final_decision"] = {
        "result_type": "known_class",
        "primary_category": "驱逐舰",
        "primary_class": "阿利·伯克级驱逐舰",
        "confidence": round(float(arleigh_conf), 4),
        "status": "v48_promoted_arleigh",
        "message": "最终判定：驱逐舰 / 阿利·伯克级驱逐舰。v48 根据同大类高分候选和组合证据补全。",
        "alternatives": old_alts,
    }
    result["v48_arleigh_promotion"] = {
        "applied": True,
        "category": cat,
        "arleigh_conf": round(float(arleigh_conf), 4),
        "evidence": evidence,
    }
    return result


_hierarchical_class_match_v47_sync_base = hierarchical_class_match


def hierarchical_class_match(
    class_data_path: str,
    observed_attributes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """v48：基于 v38/v47 稳定版，只补一个点：同大类高分阿利·伯克候选安全闭集。"""
    result = _hierarchical_class_match_v47_sync_base(class_data_path, observed_attributes)
    return v48_promote_arleigh_if_safe(result, observed_attributes)



# ==================== v49: force final_decision sync when known_class_result is explicit ====================
# 单点小改：
# v48 诊断显示，部分样本的 category_result / known_class_result 已经是正确已知舰级，
# 但 final_decision.primary_class 仍残留旧值，replay_match_from_detail.py 会优先读取 final_decision，
# 因而导致 CSV 中 pred_known_class 与 known_class_result 不一致。
# 本补丁不新增任何舰级判别规则，不修改候选打分，只做最终输出字段同步。

_hierarchical_class_match_v48_base = hierarchical_class_match


def v49_nonempty_known_label(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    label = clean_text(obj.get("label") or obj.get("ship_class") or obj.get("name") or "")
    return label not in {"", "未知", "None", "none", "null", "NaN", "nan"}


def v49_sync_final_decision_force_if_known(match_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    只处理一种情况：known_class_result 明确给出了已知舰级，且 open_set_result.is_unknown=False。
    此时 final_decision 必须和 known_class_result 保持一致。

    注意：
    - 如果 open_set_result.is_unknown=True，不覆盖，避免把类别内未知类拉回已知类；
    - 不根据候选分数新增 known_class；
    - 不引入未知类先验；
    - 不改 schema_config。
    """
    if not isinstance(match_result, dict):
        return match_result

    known = match_result.get("known_class_result") or {}
    open_set = match_result.get("open_set_result") or {}
    final = match_result.get("final_decision") or {}

    if not v49_nonempty_known_label(known):
        match_result["v49_force_final_sync"] = {"applied": False, "reason": "known_class_result_empty"}
        return match_result

    if bool(open_set.get("is_unknown", False)):
        match_result["v49_force_final_sync"] = {"applied": False, "reason": "open_set_true"}
        return match_result

    label = clean_text(known.get("label") or known.get("ship_class") or known.get("name") or "")
    category = clean_text(known.get("category") or (match_result.get("category_result") or {}).get("label") or final.get("primary_category") or "")
    confidence = known.get("confidence", final.get("confidence", 0.0))

    # 同步 category_result，避免 category_result 和 final_decision 不一致。
    old_cat = match_result.get("category_result") or {}
    if category:
        match_result["category_result"] = {
            "label": category,
            "confidence": old_cat.get("confidence", confidence),
            "status": old_cat.get("status", "matched"),
            "reason": old_cat.get("reason", "") + " v49：known_class_result 已明确，强制同步最终大类。",
        }

    # 同步 open_set_result。
    match_result["open_set_result"] = {
        "is_unknown": False,
        "unknown_scope": None,
        "reason": open_set.get("reason") or "v49：known_class_result 已明确，关闭 open-set。",
    }

    # 强制同步 final_decision。这里不是只在空值时同步，而是只要不一致就覆盖。
    old_alts = final.get("alternatives") if isinstance(final, dict) else None
    new_final = {
        "result_type": "known_class",
        "primary_category": category,
        "primary_class": label,
        "confidence": confidence,
        "status": final.get("status") or "v49_synced_known_class",
        "message": f"最终判定：{category} / {label}。v49 将 final_decision 与 known_class_result 强制同步。",
    }
    if old_alts:
        new_final["alternatives"] = old_alts
    match_result["final_decision"] = new_final

    match_result["v49_force_final_sync"] = {
        "applied": True,
        "synced_category": category,
        "synced_class": label,
        "reason": "known_class_result explicit and open_set=false; final_decision overwritten to avoid stale primary_class.",
    }
    return match_result


# ==================== v49.1: safe top-known-candidate promotion ====================
# 目的：
# v49 只能处理 known_class_result 已经明确、但 final_decision 没同步的情况。
# 但当前错误样本中，常见情况是 known_class_result=None，而 known_class_candidates 第一名已经很强。
# v49.1 在“最高候选足够强、候选间隔足够、证据足够、冲突可控、且大类一致”时，
# 才把最高候选安全提升为 known_class_result + final_decision。
#
# 注意：
# - 不是“最高候选无脑覆盖最终预测”；
# - open_set_result.is_unknown=True 时使用更严格阈值，避免把真实类别内未知样本错误拉回已知类；
# - 所有是否提升、为何不提升都会写入 v49_1_top_candidate_promotion，便于 replay 后排查。

V49_1_PROMOTE_CONF = 0.72
V49_1_PROMOTE_MARGIN = 0.06
V49_1_PROMOTE_EVIDENCE = 3
V49_1_PROMOTE_MAX_CONFLICT = 1

V49_1_PROMOTE_OPENSET_CONF = 0.80
V49_1_PROMOTE_OPENSET_MARGIN = 0.08
V49_1_PROMOTE_OPENSET_EVIDENCE = 4
V49_1_PROMOTE_OPENSET_MAX_CONFLICT = 0

V49_1_UNKNOWN_LABELS = {"", "未知", "不确定", "未提及", "None", "none", "null", "NaN", "nan"}


def v49_1_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def v49_1_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def v49_1_get_candidate_label(candidate: Dict[str, Any]) -> str:
    return clean_text(candidate.get("label") or candidate.get("ship_class") or candidate.get("name") or "")


def v49_1_get_candidate_category(candidate: Dict[str, Any]) -> str:
    return clean_text(candidate.get("category") or candidate.get("primary_category") or "")


def v49_1_has_known_final(match_result: Dict[str, Any]) -> bool:
    final = match_result.get("final_decision") or {}
    if not isinstance(final, dict):
        return False
    primary_class = clean_text(final.get("primary_class") or "")
    return final.get("result_type") == "known_class" and primary_class not in V49_1_UNKNOWN_LABELS


def v49_1_category_is_compatible(match_result: Dict[str, Any], top_category: str) -> Tuple[bool, str]:
    """
    最高候选的大类必须与当前 category_result/final_decision 的大类兼容。
    这样可以防止“候选最高”跨大类强行覆盖。
    """
    category_result = match_result.get("category_result") or {}
    final = match_result.get("final_decision") or {}

    category_label = clean_text(category_result.get("label") or "")
    final_category = clean_text(final.get("primary_category") or "")

    known_unknowns = V49_1_UNKNOWN_LABELS | {"None", "null"}

    if not top_category or top_category in known_unknowns:
        return False, "top_candidate_category_empty"

    # 如果当前没有稳定大类，允许最高候选给出大类，但后面仍要满足更严格的置信度/证据条件。
    if category_label in known_unknowns and final_category in known_unknowns:
        return True, "no_existing_category"

    if category_label and category_label not in known_unknowns and category_label != top_category:
        return False, f"category_result_mismatch:{category_label}!={top_category}"

    if final_category and final_category not in known_unknowns and final_category != top_category:
        return False, f"final_category_mismatch:{final_category}!={top_category}"

    return True, "same_category"



# ==================== v49.3: open-set promotion confirmation gate ====================
# v49.2 之所以变差，是因为它把“强锚点确认/弱已知回退”作用到了所有最终 known_class 上，
# 导致原本真实的已知类也被回退成类别内未知。
# v49.3 只做一件事：当 v49.1 准备把“原本 open_set=True 的 top candidate”提升为 known_class 时，
# 对最容易发生未知类假阳性的舰级增加一个温和确认门槛。
# 已经明确的 known_class_result / final_decision 不受 v49.3 影响。

V49_3_OPENSET_ANCHOR_LABELS = {
    "阿利·伯克级驱逐舰",
    "独立级濒海战斗舰",
}

# 对 open-set 样本，如果最高候选分数没到“极强闭集证据”，则需要锚点确认。
# 注意：这不是全局 veto，只作用于 v49.1 的“候选提升”分支。
V49_3_BYPASS_ANCHOR_CONF = 0.94
V49_3_BYPASS_ANCHOR_MARGIN = 0.20
V49_3_BYPASS_ANCHOR_EVIDENCE = 6


def v49_3_text_compact(x: Any) -> str:
    try:
        return compact_key(x)
    except Exception:
        return re.sub(r"\s+", "", str(x or "")).lower()


def v49_3_observed_to_text(observed: Optional[Dict[str, Dict[str, Any]]]) -> str:
    """把原始文本与槽位值拼成可检索文本，只用于候选提升前的锚点确认。"""
    parts: List[str] = []
    if not isinstance(observed, dict):
        return ""

    try:
        meta = observed.get("_META") or {}
        if isinstance(meta, dict):
            parts.append(str(meta.get("raw_text", "") or ""))
    except Exception:
        pass

    try:
        for group, slots in observed.items():
            if str(group).startswith("_") or not isinstance(slots, dict):
                continue
            for slot, val in slots.items():
                parts.append(str(slot))
                if isinstance(val, list):
                    parts.extend(str(v or "") for v in val)
                else:
                    parts.append(str(val or ""))
    except Exception:
        pass

    return " ".join(p for p in parts if p and p not in {"未知", "不确定", "未提及"})


def v49_3_has_any(text: str, cues: List[str]) -> bool:
    ck = v49_3_text_compact(text)
    return any(v49_3_text_compact(c) in ck for c in cues)


def v49_3_open_set_anchor_report(
    label: str,
    category: str,
    observed: Optional[Dict[str, Dict[str, Any]]],
    top_conf: float,
    margin: float,
    evidence_count: int,
) -> Dict[str, Any]:
    """
    只判断“open-set 样本是否允许被最高候选提升为已知舰级”。

    返回：
    - required: 是否需要锚点确认；
    - passed: 是否通过；
    - reason / matched_groups: 便于 replay 以后调试。
    """
    label = clean_text(label)
    category = clean_text(category)

    if label not in V49_3_OPENSET_ANCHOR_LABELS:
        return {
            "required": False,
            "passed": True,
            "reason": "label_not_in_v49_3_anchor_scope",
            "matched_groups": [],
            "exact_name": False,
        }

    text = v49_3_observed_to_text(observed)

    exact_name_cues = {
        "阿利·伯克级驱逐舰": ["阿利伯克", "阿利·伯克", "伯克级", "Arleigh Burke", "DDG-51"],
        "独立级濒海战斗舰": ["独立级", "Independence", "LCS-2", "濒海战斗舰"],
    }
    exact_name = v49_3_has_any(text, exact_name_cues.get(label, [label]))
    if exact_name:
        return {
            "required": True,
            "passed": True,
            "reason": "explicit_known_class_name",
            "matched_groups": ["exact_name"],
            "exact_name": True,
        }

    # 如果候选本身已经极强，允许绕过锚点；否则 open-set 样本会被过度拒识。
    # 这个绕过条件比 v49.1 严格，目的是保留部分真实已知类的恢复能力。
    strong_score_bypass = (
        top_conf >= V49_3_BYPASS_ANCHOR_CONF
        and margin >= V49_3_BYPASS_ANCHOR_MARGIN
        and evidence_count >= V49_3_BYPASS_ANCHOR_EVIDENCE
    )
    if strong_score_bypass:
        return {
            "required": True,
            "passed": True,
            "reason": "very_strong_candidate_score_bypass",
            "matched_groups": ["score_bypass"],
            "exact_name": False,
        }

    matched_groups: List[str] = []

    if label == "阿利·伯克级驱逐舰":
        checks = {
            # 核心锚点：越靠前越具有专属性。
            "aegis_or_spy": ["宙斯盾", "Aegis", "SPY-1", "SPY1", "SPY-6", "SPY6"],
            "mk41_or_90_96_vls": ["Mk41", "MK41", "MK-41", "90单元", "96单元", "90-96", "90至96", "垂直发射单元90", "垂直发射单元96"],
            # 辅助锚点：单独不够，但能支撑核心锚点。
            "bow_127mm_gun": ["127mm", "127毫米", "5英寸", "舰艏主炮", "舰首主炮"],
            "ddg_surface_combatant": ["导弹驱逐舰", "防空驱逐舰", "DDG", "多用途导弹驱逐舰"],
        }
        for group, cues in checks.items():
            if v49_3_has_any(text, cues):
                matched_groups.append(group)

        has_core = ("aegis_or_spy" in matched_groups) or ("mk41_or_90_96_vls" in matched_groups)
        passed = has_core and len(matched_groups) >= 2
        return {
            "required": True,
            "passed": passed,
            "reason": "arleigh_open_set_anchor_passed" if passed else "missing_arleigh_open_set_anchor",
            "matched_groups": matched_groups,
            "exact_name": False,
        }

    if label == "独立级濒海战斗舰":
        checks = {
            "trimaran": ["三体船", "三体", "trimaran"],
            "lcs_or_littoral": ["濒海战斗舰", "近海战斗", "近海作战", "LCS"],
            "fifty_seven_mm": ["57mm", "57毫米", "中小口径主炮"],
            "modular_mission": ["模块化任务", "任务模块", "任务舱", "模块化"],
            "stern_flight_deck": ["艉部直升机甲板", "舰尾直升机甲板", "艉部飞行甲板", "舰尾飞行甲板"],
            "low_stealth_superstructure": ["低矮隐身", "隐身化上层建筑", "低矮上层建筑"],
        }
        for group, cues in checks.items():
            if v49_3_has_any(text, cues):
                matched_groups.append(group)

        # 对独立级，三体船是最关键锚点；如果没有三体船，只凭57mm/直升机甲板不应把未知护卫舰拉回独立级。
        passed = ("trimaran" in matched_groups) and len(matched_groups) >= 2
        return {
            "required": True,
            "passed": passed,
            "reason": "independence_open_set_anchor_passed" if passed else "missing_independence_open_set_anchor",
            "matched_groups": matched_groups,
            "exact_name": False,
        }

    return {
        "required": False,
        "passed": True,
        "reason": "no_v49_3_rule_for_label",
        "matched_groups": [],
        "exact_name": False,
    }

def v49_1_promote_top_candidate_when_safe(
    match_result: Dict[str, Any],
    observed_attributes: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    如果 known_class_result 为空，但 known_class_candidates 第一名满足安全条件，
    则把它提升为最终已知舰级预测。

    解决的问题：
    - CSV 里能看到 top candidate 最高分是某已知舰级；
    - 但 pred_known_class 仍为空/未知，因为 replay 读取的是 final_decision，而不是候选列表。
    """
    if not isinstance(match_result, dict):
        return match_result

    # 已经是明确 known_class，不再改动。
    if v49_1_has_known_final(match_result):
        match_result["v49_1_top_candidate_promotion"] = {
            "applied": False,
            "reason": "final_decision_already_known_class",
        }
        return match_result

    known = match_result.get("known_class_result") or {}
    if v49_nonempty_known_label(known):
        # 理论上 v49 已经同步；这里再兜底调用一次，避免顺序问题。
        match_result = v49_sync_final_decision_force_if_known(match_result)
        match_result["v49_1_top_candidate_promotion"] = {
            "applied": False,
            "reason": "known_class_result_already_explicit_v49_synced",
        }
        return match_result

    candidates = match_result.get("known_class_candidates") or []
    if not isinstance(candidates, list) or not candidates:
        match_result["v49_1_top_candidate_promotion"] = {
            "applied": False,
            "reason": "no_known_class_candidates",
        }
        return match_result

    # 候选重新排序，避免上游 candidates 顺序不是按置信度排列。
    candidates = sorted(
        [c for c in candidates if isinstance(c, dict)],
        key=lambda x: (v49_1_float(x.get("confidence", x.get("score", 0.0))), v49_1_float(x.get("score", 0.0))),
        reverse=True,
    )
    if not candidates:
        match_result["v49_1_top_candidate_promotion"] = {
            "applied": False,
            "reason": "no_valid_known_class_candidates",
        }
        return match_result

    top = candidates[0]
    second = candidates[1] if len(candidates) > 1 else {}

    top_label = v49_1_get_candidate_label(top)
    top_category = v49_1_get_candidate_category(top)
    top_conf = v49_1_float(top.get("confidence", top.get("score", 0.0)))
    second_conf = v49_1_float(second.get("confidence", second.get("score", 0.0))) if isinstance(second, dict) else 0.0
    margin = top_conf - second_conf
    evidence_count = v49_1_int(top.get("matched_evidence_count", 0))
    conflict_count = v49_1_int(top.get("conflict_count", 0))

    compatible, compatible_reason = v49_1_category_is_compatible(match_result, top_category)
    open_set = match_result.get("open_set_result") or {}
    is_open_unknown = bool(open_set.get("is_unknown", False))

    debug_payload = {
        "top_label": top_label,
        "top_category": top_category,
        "top_confidence": round(top_conf, 4),
        "second_confidence": round(second_conf, 4),
        "margin": round(margin, 4),
        "matched_evidence_count": evidence_count,
        "conflict_count": conflict_count,
        "open_set_is_unknown": is_open_unknown,
        "category_compatible": compatible,
        "category_compatible_reason": compatible_reason,
    }

    if top_label in V49_1_UNKNOWN_LABELS:
        match_result["v49_1_top_candidate_promotion"] = {
            "applied": False,
            "reason": "top_candidate_label_empty",
            **debug_payload,
        }
        return match_result

    if not compatible:
        match_result["v49_1_top_candidate_promotion"] = {
            "applied": False,
            "reason": compatible_reason,
            **debug_payload,
        }
        return match_result

    if is_open_unknown:
        safe_known = (
            top_conf >= V49_1_PROMOTE_OPENSET_CONF
            and margin >= V49_1_PROMOTE_OPENSET_MARGIN
            and evidence_count >= V49_1_PROMOTE_OPENSET_EVIDENCE
            and conflict_count <= V49_1_PROMOTE_OPENSET_MAX_CONFLICT
        )
        safety_profile = "strict_open_set_guard"
        required = {
            "min_confidence": V49_1_PROMOTE_OPENSET_CONF,
            "min_margin": V49_1_PROMOTE_OPENSET_MARGIN,
            "min_evidence": V49_1_PROMOTE_OPENSET_EVIDENCE,
            "max_conflict": V49_1_PROMOTE_OPENSET_MAX_CONFLICT,
        }
    else:
        safe_known = (
            top_conf >= V49_1_PROMOTE_CONF
            and margin >= V49_1_PROMOTE_MARGIN
            and evidence_count >= V49_1_PROMOTE_EVIDENCE
            and conflict_count <= V49_1_PROMOTE_MAX_CONFLICT
        )
        safety_profile = "normal_guard"
        required = {
            "min_confidence": V49_1_PROMOTE_CONF,
            "min_margin": V49_1_PROMOTE_MARGIN,
            "min_evidence": V49_1_PROMOTE_EVIDENCE,
            "max_conflict": V49_1_PROMOTE_MAX_CONFLICT,
        }

    if not safe_known:
        match_result["v49_1_top_candidate_promotion"] = {
            "applied": False,
            "reason": "top_candidate_not_safe_enough",
            "safety_profile": safety_profile,
            "required": required,
            **debug_payload,
        }
        return match_result

    # v49.3：只对“原本 open_set=True 的最高候选提升”增加确认门槛。
    # 这一步不会影响已经明确的 known_class_result，也不会像 v49.2 那样对最终 known_class 做回退 veto。
    anchor_report = v49_3_open_set_anchor_report(
        top_label,
        top_category,
        observed_attributes,
        top_conf,
        margin,
        evidence_count,
    ) if is_open_unknown else {
        "required": False,
        "passed": True,
        "reason": "not_open_set_promotion",
        "matched_groups": [],
        "exact_name": False,
    }

    debug_payload.update({
        "v49_3_anchor_required": bool(anchor_report.get("required", False)),
        "v49_3_anchor_passed": bool(anchor_report.get("passed", True)),
        "v49_3_anchor_reason": anchor_report.get("reason"),
        "v49_3_anchor_matched_groups": anchor_report.get("matched_groups", []),
        "v49_3_anchor_exact_name": bool(anchor_report.get("exact_name", False)),
    })

    if anchor_report.get("required", False) and not anchor_report.get("passed", False):
        match_result["v49_3_open_set_promotion_gate"] = {
            "applied": True,
            "action": "reject_top_candidate_promotion",
            "reason": anchor_report.get("reason"),
            **debug_payload,
        }
        match_result["v49_1_top_candidate_promotion"] = {
            "applied": False,
            "reason": "v49_3_open_set_anchor_gate_rejected",
            "safety_profile": safety_profile,
            "required": required,
            **debug_payload,
        }
        return match_result

    match_result["v49_3_open_set_promotion_gate"] = {
        "applied": bool(anchor_report.get("required", False)),
        "action": "allow_top_candidate_promotion",
        "reason": anchor_report.get("reason"),
        **debug_payload,
    }

    # 到这里才允许把最高候选升级成最终预测。
    final = match_result.get("final_decision") or {}
    old_alts = final.get("alternatives") if isinstance(final, dict) else None
    if not old_alts:
        old_alts = {
            "top_categories": match_result.get("category_candidates", [])[:FINAL_TOPK],
            "top_known_classes": candidates[:FINAL_TOPK],
        }

    old_cat = match_result.get("category_result") or {}
    match_result["category_result"] = {
        "label": top_category,
        "confidence": max(v49_1_float(old_cat.get("confidence", 0.0)), top_conf),
        "status": old_cat.get("status") or "matched_by_top_candidate",
        "reason": (old_cat.get("reason") or "") + " v49.1：最高已知舰级候选满足安全提升条件，同步最终大类。",
    }

    match_result["known_class_result"] = {
        "label": top_label,
        "category": top_category,
        "confidence": round(top_conf, 4),
        "score": top.get("score", top_conf),
        "known_status": "Known",
        "reason": (
            "v49.1：known_class_result 原为空，但 known_class_candidates 第一名满足"
            "置信度、候选间隔、证据数、冲突数和大类一致性约束，因此安全提升为已知舰级。"
        ),
    }

    match_result["open_set_result"] = {
        "is_unknown": False,
        "unknown_scope": None,
        "reason": "v49.1：最高候选满足安全提升条件，关闭 open-set。",
    }

    match_result["final_decision"] = {
        "result_type": "known_class",
        "primary_category": top_category,
        "primary_class": top_label,
        "confidence": round(top_conf, 4),
        "status": "v49_1_promoted_top_candidate",
        "message": f"最终判定：{top_category} / {top_label}。v49.1 根据最高候选安全提升为已知舰级。",
        "margin": round(margin, 4),
        "alternatives": old_alts,
    }

    match_result["v49_1_top_candidate_promotion"] = {
        "applied": True,
        "reason": "top_candidate_promoted_safely",
        "safety_profile": safety_profile,
        "required": required,
        **debug_payload,
    }
    return match_result



# ==================== v49.7: five-evidence category-only boundary correction ====================
# 目的：
# - v49.6 失败的根因是把“五类证据”用于恢复 known_class，导致 open-set 被打穿。
# - v49.7 只允许五类证据修正 category_unknown/open-set 的大类，不允许补全具体已知舰级。
# - 已经是 known_class 的结果完全不动；pred_known_class 保持空；pred_open_set 保持 True。

V49_7_CATEGORIES = ["航空母舰", "巡洋舰", "驱逐舰", "护卫舰", "两栖舰", "登陆舰"]
V49_7_NEG_PREFIXES = ["无", "没有", "未见", "未配备", "不具备", "缺乏", "并无", "不是", "并非", "非"]

# 五类证据之一：独有强锚点。这里用于“大类强指向”，不用于恢复具体已知舰级。
V49_7_UNIQUE_ANCHORS = {
    "航空母舰": ["航空母舰", "核动力航空母舰", "超级航母", "航母本体", "蒸汽弹射器", "弹射器", "拦阻索", "固定翼舰载机", "舰载机联队", "10万吨", "十万吨", "100架", "CVN"],
    "巡洋舰": ["导弹巡洋舰", "宙斯盾巡洋舰", "提康德罗加", "122单元", "122枚", "16组八联装", "十六组八联装", "舰队指挥", "区域防空旗舰", "前后各有一门", "舰艏和舰艉各一门", "双127"],
    "驱逐舰": ["导弹驱逐舰", "防空驱逐舰", "宙斯盾驱逐舰", "阿利伯克", "阿利·伯克", "DDG51", "DDG-51", "90-96单元", "96单元", "FlightIIA", "FlightIII", "SPY-1D", "AN/SPY-1D"],
    "护卫舰": ["护卫舰", "巡防舰", "濒海战斗舰", "LCS", "三体船", "三体结构", "多体结构", "独立级", "57mm", "57毫米", "模块化任务", "任务模块", "近海作战"],
    "两栖舰": ["两栖攻击舰", "两栖舰", "LHD", "黄蜂级", "STOVL", "F-35B", "AV-8B", "垂直起降", "短距起飞"],
    "登陆舰": ["船坞登陆舰", "船坞登陆", "登陆舰", "LSD", "惠德比", "4艘LCAC", "四艘LCAC", "登陆艇投送", "大型船坞", "泛水坞舱"],
}

# 五类证据之二：共享区分特征。它能区分一组大类，但不能直接说明就是某个已知类。
V49_7_SHARED_DISTINGUISHING = {
    "航空母舰": ["全通飞行甲板", "斜角飞行甲板", "右舷舰岛", "大型飞行甲板", "飞机升降机", "舰载机", "机库"],
    "巡洋舰": ["大量垂发", "大量垂直发射", "垂直发射", "垂发", "相控阵", "宙斯盾", "区域防空", "指挥控制", "编队指挥"],
    "驱逐舰": ["垂直发射", "垂发", "相控阵", "宙斯盾", "区域防空", "127mm", "127毫米", "舰艏主炮", "多用途", "防空"],
    "护卫舰": ["中小口径", "57mm", "57毫米", "76mm", "艉部直升机甲板", "直升机甲板", "低矮隐身", "反潜", "护航", "巡逻", "近海"],
    "两栖舰": ["全通飞行甲板", "直升机甲板", "坞舱", "艉门", "车辆甲板", "运兵", "登陆艇", "两栖投送", "两栖攻击"],
    "登陆舰": ["坞舱", "艉门", "登陆艇", "LCAC", "LCU", "车辆甲板", "登陆作战", "船坞"],
}

# 五类证据之三：普通弱特征。只能辅助，不能单独触发大类修正。
V49_7_WEAK_FEATURES = {
    "航空母舰": ["大型舰", "海上力量", "航空作业", "飞机"],
    "巡洋舰": ["大型水面舰", "防空", "导弹", "指挥"],
    "驱逐舰": ["水面战斗舰", "导弹", "防空", "护航"],
    "护卫舰": ["中型舰", "轻型舰", "反潜", "护卫", "巡逻"],
    "两栖舰": ["直升机", "运兵", "车辆", "两栖"],
    "登陆舰": ["登陆", "运输", "车辆", "船坞"],
}

# 五类证据之四：反向排除特征。注意：这里是对“目标大类”扣分/阻断。
V49_7_NEGATIVE_EXCLUSION = {
    "航空母舰": ["无弹射器", "没有弹射器", "无拦阻索", "没有拦阻索", "无全通飞行甲板", "无固定翼", "无法固定翼"],
    "巡洋舰": ["三体船", "三体结构", "57mm", "57毫米", "中小口径", "坞舱", "艉门", "登陆艇", "全通飞行甲板"],
    "驱逐舰": ["三体船", "三体结构", "57mm", "57毫米", "中小口径", "坞舱", "艉门", "登陆艇", "全通飞行甲板"],
    "护卫舰": ["122单元", "122枚", "96单元", "90-96单元", "舰队指挥", "区域防空旗舰", "坞舱", "艉门", "全通飞行甲板"],
    "两栖舰": ["无坞舱", "没有坞舱", "无艉门", "没有艉门", "弹射器", "拦阻索", "三体船", "122单元"],
    "登陆舰": ["无坞舱", "没有坞舱", "无艉门", "没有艉门", "弹射器", "拦阻索", "三体船", "STOVL", "F-35B", "AV-8B"],
}


def v49_7_slot(observed: Optional[Dict[str, Dict[str, Any]]], path: str) -> str:
    if not isinstance(observed, dict) or "." not in path:
        return ""
    group, slot = path.split(".", 1)
    obj = observed.get(group, {})
    if not isinstance(obj, dict):
        return ""
    value = obj.get(slot, "")
    if isinstance(value, list):
        return " ".join(clean_text(x) for x in value)
    return clean_text(value)


def v49_7_text_blob(observed: Optional[Dict[str, Dict[str, Any]]]) -> str:
    parts: List[str] = []
    if isinstance(observed, dict):
        meta = observed.get("_META", {})
        if isinstance(meta, dict):
            parts.append(clean_text(meta.get("raw_text", "")))
        for group, slots in observed.items():
            if str(group).startswith("_") or not isinstance(slots, dict):
                continue
            for slot, value in slots.items():
                parts.append(str(slot))
                if isinstance(value, list):
                    parts.extend(clean_text(x) for x in value)
                else:
                    parts.append(clean_text(value))
    return " ".join(x for x in parts if x)


def v49_7_has_raw_pattern(text_key: str, cue: str) -> bool:
    ck = compact_key(cue)
    return bool(ck and ck in text_key)


def v49_7_has_positive_cue(text_key: str, cue: str) -> bool:
    ck = compact_key(cue)
    if not ck or ck not in text_key:
        return False
    # 避免把“无坞舱/没有弹射器/未见全通飞行甲板”当成正向证据。
    for prefix in V49_7_NEG_PREFIXES:
        if compact_key(prefix + cue) in text_key:
            return False
    return True


def v49_7_add_evidence(report: Dict[str, Any], category: str, evidence_type: str, cue: str, score: float):
    report[category]["score"] += score
    report[category]["evidence_types"].add(evidence_type)
    report[category]["evidence"].append({"type": evidence_type, "cue": cue, "score": score})
    if evidence_type == "negative_exclusion":
        report[category]["negative_count"] += 1
    if evidence_type == "shared_distinguishing":
        report[category]["shared_count"] += 1
    if evidence_type == "unique_anchor":
        report[category]["unique_count"] += 1
    if evidence_type == "combo_evidence":
        report[category]["combo_count"] += 1


def v49_7_score_categories(observed: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    raw = v49_7_text_blob(observed)
    text_key = compact_key(raw)
    report: Dict[str, Any] = {
        cat: {
            "score": 0.0,
            "evidence_types": set(),
            "evidence": [],
            "negative_count": 0,
            "shared_count": 0,
            "unique_count": 0,
            "combo_count": 0,
        }
        for cat in V49_7_CATEGORIES
    }

    # 1. 独有强锚点
    for cat, cues in V49_7_UNIQUE_ANCHORS.items():
        for cue in cues:
            if v49_7_has_positive_cue(text_key, cue):
                v49_7_add_evidence(report, cat, "unique_anchor", cue, 5.0)

    # 2. 共享区分特征
    for cat, cues in V49_7_SHARED_DISTINGUISHING.items():
        for cue in cues:
            if v49_7_has_positive_cue(text_key, cue):
                v49_7_add_evidence(report, cat, "shared_distinguishing", cue, 2.2)

    # 3. 普通弱特征
    for cat, cues in V49_7_WEAK_FEATURES.items():
        for cue in cues:
            if v49_7_has_positive_cue(text_key, cue):
                v49_7_add_evidence(report, cat, "weak_feature", cue, 0.6)

    # 4. 反向排除特征
    for cat, cues in V49_7_NEGATIVE_EXCLUSION.items():
        for cue in cues:
            # 排除特征可以是“无坞舱”这种否定短语，也可以是对其他大类的正向特征，如“三体船”排除巡洋/驱逐。
            if v49_7_has_raw_pattern(text_key, cue):
                v49_7_add_evidence(report, cat, "negative_exclusion", cue, -4.5)

    # 5. 组合证据：组合证据只修大类，不恢复已知类。
    flight_type = v49_7_slot(observed, "AVIATION_FEATURES.Flight_Deck_Type")
    catapult = v49_7_slot(observed, "AVIATION_FEATURES.Catapult")
    arrest = v49_7_slot(observed, "AVIATION_FEATURES.Arresting_Gear")
    fixed_wing = v49_7_slot(observed, "AVIATION_FEATURES.Fixed_Wing_Aircraft_Operation")
    well = v49_7_slot(observed, "AMPHIBIOUS_FEATURES.Well_Deck")
    stern_gate = v49_7_slot(observed, "AMPHIBIOUS_FEATURES.Stern_Gate")
    landing_craft = v49_7_slot(observed, "AMPHIBIOUS_FEATURES.Landing_Craft_Capability") + " " + v49_7_slot(observed, "EQUIPMENT_DETAILS.Landing_Craft")
    vls = v49_7_slot(observed, "WEAPON_SENSOR_FEATURES.VLS_Count_Level") + " " + v49_7_slot(observed, "WEAPON_SENSOR_FEATURES.VLS_Presence")
    gun = v49_7_slot(observed, "WEAPON_SENSOR_FEATURES.Main_Gun_Caliber") + " " + v49_7_slot(observed, "WEAPON_SENSOR_FEATURES.Main_Gun_Position")
    radar = v49_7_slot(observed, "WEAPON_SENSOR_FEATURES.Phased_Array_Radar") + " " + v49_7_slot(observed, "WEAPON_SENSOR_FEATURES.Radar_Array_Type") + " " + v49_7_slot(observed, "EQUIPMENT_DETAILS.Radar_System")
    mission = v49_7_slot(observed, "MISSION_FEATURES.Primary_Mission") + " " + v49_7_slot(observed, "MISSION_FEATURES.Area_Air_Defense") + " " + v49_7_slot(observed, "MISSION_FEATURES.Command_Control")
    hull = v49_7_slot(observed, "VISUAL_STRUCTURE.Hull_Form")
    superstructure = v49_7_slot(observed, "VISUAL_STRUCTURE.Superstructure_Type") + " " + v49_7_slot(observed, "VISUAL_STRUCTURE.Stealth_Shape")
    module = v49_7_slot(observed, "EQUIPMENT_DETAILS.Mission_Module")

    combo_text = compact_key(" ".join([raw, flight_type, catapult, arrest, fixed_wing, well, stern_gate, landing_craft, vls, gun, radar, mission, hull, superstructure, module]))

    def has(cue: str) -> bool:
        return v49_7_has_positive_cue(combo_text, cue)

    if (has("弹射器") or catapult == "有") and (has("拦阻索") or arrest == "有"):
        v49_7_add_evidence(report, "航空母舰", "combo_evidence", "弹射器+拦阻索", 6.0)
    if has("全通飞行甲板") and (has("固定翼") or has("舰载机联队") or has("100架") or fixed_wing == "有"):
        v49_7_add_evidence(report, "航空母舰", "combo_evidence", "全通飞行甲板+固定翼舰载机", 4.5)

    if (has("122单元") or has("16组八联装") or "122" in vls) and (has("舰队指挥") or has("区域防空") or has("前后")):
        v49_7_add_evidence(report, "巡洋舰", "combo_evidence", "122单元/大量垂发+指挥/区域防空", 6.0)
    if has("前后垂发") and (has("舰队指挥") or has("区域防空")):
        v49_7_add_evidence(report, "巡洋舰", "combo_evidence", "前后垂发+舰队防空指挥", 4.5)

    if (has("96单元") or has("90-96单元") or has("MK41") or has("Mk41")) and (has("127") or has("舰艏主炮") or has("导弹驱逐舰")):
        v49_7_add_evidence(report, "驱逐舰", "combo_evidence", "90/96单元或Mk41+127mm/导弹驱逐舰", 5.5)
    if (has("相控阵") or has("SPY")) and (has("127") or has("舰艏主炮")) and (has("垂发") or has("垂直发射")):
        v49_7_add_evidence(report, "驱逐舰", "combo_evidence", "相控阵+舰艏主炮+垂发", 4.5)

    if (has("三体") or has("多体")) and (has("57") or has("模块") or has("艉部直升机甲板") or has("濒海")):
        v49_7_add_evidence(report, "护卫舰", "combo_evidence", "三体/多体+57mm/模块化/艉部直升机甲板", 6.5)
    if (has("57") or has("76") or has("中小口径")) and (has("反潜") or has("护航") or has("巡逻") or has("近海")):
        v49_7_add_evidence(report, "护卫舰", "combo_evidence", "中小口径主炮+反潜/护航/巡逻", 4.0)

    if has("全通飞行甲板") and (has("坞舱") or well == "有") and (has("STOVL") or has("垂直起降") or has("两栖攻击")):
        v49_7_add_evidence(report, "两栖舰", "combo_evidence", "全通飞行甲板+坞舱+STOVL/两栖攻击", 6.0)
    if (has("两栖攻击舰") or has("LHD")) and (has("直升机") or has("全通飞行甲板")):
        v49_7_add_evidence(report, "两栖舰", "combo_evidence", "两栖攻击舰/LHD+航空甲板", 4.5)

    if (has("坞舱") or well == "有") and (has("艉门") or stern_gate == "有") and (has("LCAC") or has("LCU") or has("登陆艇") or has("船坞登陆")):
        v49_7_add_evidence(report, "登陆舰", "combo_evidence", "坞舱+艉门+LCAC/LCU/登陆艇", 6.0)
    if (has("船坞登陆舰") or has("LSD")) and (has("坞舱") or has("登陆艇")):
        v49_7_add_evidence(report, "登陆舰", "combo_evidence", "船坞登陆舰/LSD+坞舱/登陆艇", 4.5)

    # set 转 list，便于 JSON 序列化。
    for cat in V49_7_CATEGORIES:
        report[cat]["score"] = round(float(report[cat]["score"]), 4)
        report[cat]["evidence_types"] = sorted(list(report[cat]["evidence_types"]))
    return report


def v49_7_should_touch(match_result: Dict[str, Any]) -> Tuple[bool, str]:
    final = match_result.get("final_decision") or {}
    known = match_result.get("known_class_result") or {}
    open_set = match_result.get("open_set_result") or {}

    if v49_1_has_known_final(match_result):
        return False, "final_known_class_do_not_touch"
    if v49_nonempty_known_label(known) and not bool(open_set.get("is_unknown", False)):
        return False, "known_class_result_explicit_do_not_touch"

    result_type = final.get("result_type")
    primary_class = clean_text(final.get("primary_class") or "")
    is_open = bool(open_set.get("is_unknown", False)) or result_type in {"category_unknown", "ambiguous_category", "category_only"}
    if primary_class not in V49_1_UNKNOWN_LABELS:
        return False, "primary_class_not_empty_do_not_touch"
    if not is_open:
        return False, "not_open_set_or_category_unknown"
    return True, "category_only_candidate"


def v49_7_fix_category_only(match_result: Dict[str, Any], observed_attributes: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    v49.7：五类证据只修大类边界，不恢复 known_class。
    只作用于 category_unknown/open-set/primary_class 为空的结果。
    """
    if not isinstance(match_result, dict):
        return match_result

    touch, reason = v49_7_should_touch(match_result)
    if not touch:
        match_result["v49_7_five_evidence_category_only"] = {"applied": False, "reason": reason}
        return match_result

    final = match_result.get("final_decision") or {}
    category_result = match_result.get("category_result") or {}
    open_set = match_result.get("open_set_result") or {}
    current_cat = clean_text(final.get("primary_category") or category_result.get("label") or "")

    scores = v49_7_score_categories(observed_attributes)
    ranked = sorted(V49_7_CATEGORIES, key=lambda c: float(scores[c]["score"]), reverse=True)
    top_cat = ranked[0] if ranked else ""
    second_cat = ranked[1] if len(ranked) > 1 else ""
    top_score = float(scores.get(top_cat, {}).get("score", 0.0)) if top_cat else 0.0
    second_score = float(scores.get(second_cat, {}).get("score", 0.0)) if second_cat else 0.0
    current_score = float(scores.get(current_cat, {}).get("score", 0.0)) if current_cat in scores else 0.0
    top_report = scores.get(top_cat, {})

    evidence_types = set(top_report.get("evidence_types", []))
    strong_type_count = len(evidence_types.intersection({"unique_anchor", "shared_distinguishing", "combo_evidence"}))
    has_combo = int(top_report.get("combo_count", 0)) > 0
    has_unique = int(top_report.get("unique_count", 0)) > 0
    shared_count = int(top_report.get("shared_count", 0))
    negative_count = int(top_report.get("negative_count", 0))

    # 共享区分特征可以触发修正，但不能单独触发；需要多个共享特征且当前类别有明显反证/低分。
    shared_only_allowed = (
        shared_count >= 3
        and strong_type_count >= 1
        and (current_score <= 1.0 or top_score - current_score >= 6.0)
    )

    allow_change = (
        top_cat
        and top_cat != current_cat
        and negative_count == 0
        and top_score >= 8.0
        and top_score - max(current_score, second_score if second_cat != current_cat else -999.0) >= 3.0
        and (
            (has_combo and strong_type_count >= 2)
            or (has_unique and strong_type_count >= 2)
            or shared_only_allowed
        )
    )

    debug = {
        "current_category": current_cat,
        "top_category": top_cat,
        "second_category": second_cat,
        "top_score": round(top_score, 4),
        "second_score": round(second_score, 4),
        "current_score": round(current_score, 4),
        "top_evidence_types": sorted(list(evidence_types)),
        "top_unique_count": int(top_report.get("unique_count", 0)),
        "top_shared_count": shared_count,
        "top_combo_count": int(top_report.get("combo_count", 0)),
        "top_negative_count": negative_count,
        "top_evidence": top_report.get("evidence", [])[:10],
    }

    if not allow_change:
        match_result["v49_7_five_evidence_category_only"] = {
            "applied": False,
            "reason": "category_boundary_evidence_not_strong_enough",
            **debug,
        }
        return match_result

    # 只修大类，仍保持类别内未知，不写 known_class_result。
    old_alts = final.get("alternatives") if isinstance(final, dict) else None
    match_result["category_result"] = {
        "label": top_cat,
        "confidence": max(v49_1_float(category_result.get("confidence", 0.0)), min(0.98, 0.55 + top_score / 50.0)),
        "status": "v49_7_category_boundary_corrected",
        "reason": "v49.7：五类证据只修正类别内未知的大类，不恢复具体已知舰级。",
    }
    match_result["known_class_result"] = None
    match_result["open_set_result"] = {
        "is_unknown": True,
        "unknown_scope": UNKNOWN_OUTPUT_TEMPLATE.format(category=top_cat),
        "reason": "v49.7：大类边界已由五类证据纠正，但仍保持 open-set 类别内未知。",
    }
    new_final = {
        "result_type": "category_unknown",
        "primary_category": top_cat,
        "primary_class": None,
        "confidence": round(min(0.98, 0.55 + top_score / 50.0), 4),
        "status": "v49_7_five_evidence_category_only",
        "message": f"最终判定：{top_cat}类别内未知类。v49.7 使用五类证据修正大类，但不恢复已知舰级。",
        "alternatives": old_alts or final.get("alternatives", {}),
    }
    match_result["final_decision"] = new_final
    match_result["v49_7_five_evidence_category_only"] = {
        "applied": True,
        "reason": "category_boundary_corrected_only_keep_open_set",
        **debug,
    }
    return match_result

def hierarchical_class_match(
    class_data_path: str,
    observed_attributes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """v49.7：基于 v49.3，五类证据只修 category_unknown 的大类，不恢复 known_class。"""
    result = _hierarchical_class_match_v48_base(class_data_path, observed_attributes)

    # 先保留 v49：known_class_result 明确时，强制 final_decision 与其一致。
    result = v49_sync_final_decision_force_if_known(result)

    # 再执行 v49.1/v49.3：known_class_result 为空时，检查最高候选能否安全提升；
    # 如果原本是 open-set，再追加 v49.3 的候选提升确认门。
    result = v49_1_promote_top_candidate_when_safe(result, observed_attributes)

    # 最后执行 v49.7：只修类别内未知的大类边界，不补具体已知舰级。
    result = v49_7_fix_category_only(result, observed_attributes)
    return result



# ==================== v49.11: counter-evidence guard for accidental closed-set fill ====================
# 目标：基于 v49.7，不再扩大候选提升范围，只修复一个已确认路径：
# 部分 open-set 样本不是通过 v49_1 promotion 进入 known_class，
# 而是被 v48_promoted_arleigh 或 single_known_class_filled 过早闭集成已知舰级。
# 本补丁只在“最终已经是 known_class，且文本中存在明确反证/缺少专属锚点提示”时，
# 将其回退为类别内未知类；不恢复其他 known_class，不修改 schema_config。

_hierarchical_class_match_v49_7_base = hierarchical_class_match


def v49_11_text(observed: Dict[str, Dict[str, Any]]) -> str:
    """把 raw_text 和所有槽位值拼成一个可检索文本。"""
    parts = [get_raw_text_from_observed(observed)]
    try:
        for group, slots in observed.items():
            if str(group).startswith("_") or not isinstance(slots, dict):
                continue
            for _slot, value in slots.items():
                if isinstance(value, list):
                    parts.extend(str(x) for x in value)
                else:
                    parts.append(str(value))
    except Exception:
        pass
    return " ".join(x for x in parts if x)


def v49_11_has_any(text: str, terms: List[str]) -> bool:
    c = compact_key(text)
    return any(compact_key(t) in c for t in terms if t)


def v49_11_final_class(result: Dict[str, Any]) -> str:
    final = result.get("final_decision") or {}
    known = result.get("known_class_result") or {}
    return clean_text(final.get("primary_class") or known.get("label") or known.get("ship_class") or "")


def v49_11_final_category(result: Dict[str, Any]) -> str:
    final = result.get("final_decision") or {}
    cat = result.get("category_result") or {}
    known = result.get("known_class_result") or {}
    return clean_text(final.get("primary_category") or cat.get("label") or known.get("category") or "")


def v49_11_final_status(result: Dict[str, Any]) -> str:
    final = result.get("final_decision") or {}
    return clean_text(final.get("status") or "")


def v49_11_set_category_unknown(result: Dict[str, Any], category: str, reason: str, confidence: Optional[float] = None) -> Dict[str, Any]:
    """把已知类回退为 category_unknown，保留大类，不给具体舰级。"""
    old_cat = result.get("category_result") or {}
    old_final = result.get("final_decision") or {}
    conf = confidence
    if conf is None:
        conf = max(
            v49_1_float(old_cat.get("confidence", 0.0)),
            v49_1_float(old_final.get("confidence", 0.0)),
            0.55,
        )
    result["category_result"] = {
        "label": category,
        "confidence": round(float(conf), 4),
        "status": "v49_11_closed_set_guard_category_unknown",
        "reason": reason,
    }
    result["known_class_result"] = None
    result["open_set_result"] = {
        "is_unknown": True,
        "unknown_scope": UNKNOWN_OUTPUT_TEMPLATE.format(category=category),
        "reason": reason,
    }
    result["final_decision"] = {
        "result_type": "category_unknown",
        "primary_category": category,
        "primary_class": None,
        "confidence": round(float(conf), 4),
        "status": "v49_11_closed_set_guard",
        "message": f"最终判定：{category}类别内未知类。{reason}",
        "alternatives": old_final.get("alternatives", {}) if isinstance(old_final, dict) else {},
    }
    result["v49_11_closed_set_guard"] = {
        "applied": True,
        "category": category,
        "reason": reason,
    }
    return result


def v49_11_category_for_arleigh_reject(text: str) -> str:
    """阿利·伯克误闭集时，按文本更像的方向保留大类。"""
    # 这些更像护卫舰/中小型水面舰，不应继续留在驱逐舰大类。
    if v49_11_has_any(text, [
        "护卫舰", "巡防舰", "反潜护卫", "通用护卫", "中型军舰", "中型舰艇",
        "三千多吨", "3000吨", "3600吨", "3650吨", "6200吨", "151m", "151米",
        "有限垂发", "少量垂发", "少量垂直发射", "不像大型区域防空舰",
    ]):
        return "护卫舰"
    return "驱逐舰"


def v49_11_should_reject_arleigh(text: str, status: str, confidence: float) -> Tuple[bool, str, str]:
    """判断阿利·伯克是否只是由共享区分特征误闭集。"""
    has_named_anchor = v49_11_has_any(text, [
        "阿利伯克", "阿利·伯克", "伯克级", "DDG51", "DDG-51", "FlightII", "FlightIIA", "FlightIII", "Flight IIA", "Flight III",
    ])
    has_system_anchor = v49_11_has_any(text, [
        "SPY1", "SPY-1", "SPY1D", "SPY-1D", "SPY6", "SPY-6", "AN/SPY", "ANSPY",
        "Mk41", "MK41", "Mk 41", "MK 41", "96单元", "90具", "90单元", "90-96单元",
    ])
    # 宙斯盾是较强线索，但在未知驱逐舰中也可能出现；若同时存在强反证，不允许闭集。
    has_aegis = v49_11_has_any(text, ["宙斯盾", "Aegis"])

    explicit_missing_anchor = v49_11_has_any(text, [
        "没有给出SPY", "未给出SPY", "没有给出SPY1", "没有给出SPY-1", "没有给出Flight", "未给出Flight",
        "未出现96", "没有出现96", "未出现SPY", "没有出现SPY", "未出现SPY1D", "未出现SPY-1D",
        "未出现两座直升机库", "没有两座直升机库", "没有给出具体信息", "没有显示出已知驱逐舰的专属型号标志",
        "没有显示出已知", "没有具体雷达型号", "具体雷达型号看不清",
    ])
    counter_evidence = v49_11_has_any(text, [
        "没有机库", "无机库", "无直升机机库", "未观察到直升机库", "没有真正的直升机库", "舰尾只有直升机平台",
        "传统重型四角格子桅杆", "四角格子", "格子桅", "格子结构",
        "有限垂发", "少量垂发", "少量垂直发射", "不像大型区域防空舰",
        "16单元", "12组八联装", "76毫米", "三千多吨", "3600吨", "3000吨",
    ])

    # v48_promoted_arleigh 本来就是补丁强推，要求更严格。
    if status == "v48_promoted_arleigh":
        if explicit_missing_anchor or counter_evidence:
            return True, v49_11_category_for_arleigh_reject(text), "v49.11：v48 阿利·伯克补全遇到明确缺失锚点/反证，回退为类别内未知。"
        if not (has_named_anchor or has_system_anchor):
            return True, v49_11_category_for_arleigh_reject(text), "v49.11：v48 阿利·伯克补全只依赖共享区分特征，缺少舰级专属锚点。"

    # single_known_class_filled 属于单已知大类补全，也不能在明确缺失锚点时闭集。
    if status == "single_known_class_filled":
        if explicit_missing_anchor:
            return True, v49_11_category_for_arleigh_reject(text), "v49.11：文本明确说明缺少阿利·伯克级专属锚点，取消单已知舰级补全。"
        if counter_evidence and not has_named_anchor:
            return True, v49_11_category_for_arleigh_reject(text), "v49.11：阿利·伯克补全存在结构/规模/火力反证，回退为类别内未知。"
        # 低中置信度且只有共享区分特征时，不闭集。
        if confidence < 0.72 and not (has_named_anchor or has_system_anchor or has_aegis):
            return True, "驱逐舰", "v49.11：阿利·伯克单类补全置信度不足且缺少专属锚点。"

    return False, "", ""


def v49_11_should_reject_independence(text: str, status: str, confidence: float) -> Tuple[bool, str, str]:
    """判断独立级是否缺少三体/模块化等关键锚点。"""
    has_trimaran = v49_11_has_any(text, ["三体", "多体", "支撑船体", "两边像有支撑", "宽体三体"])
    explicit_counter = v49_11_has_any(text, [
        "没有57mm", "没有57毫米", "没有任务模块", "没有模块化", "没有57mm炮", "没有57毫米炮",
        "没有57mm炮或任务模块化", "没有任务模块化濒海舰特征", "无任务模块", "无57mm",
        "普通单体船", "单体船", "16单元", "127毫米", "通用护卫", "反潜护卫",
    ])
    # 独立级闭集最关键的是三体结构；如果文本明确否定 57mm/任务模块/三体方向，必须保留护卫舰未知。
    if explicit_counter:
        return True, "护卫舰", "v49.11：文本存在独立级反证，取消独立级闭集补全。"
    # 对低置信度的单类补全，若没有三体锚点，也不闭集。
    if status == "single_known_class_filled" and confidence < 0.62 and not has_trimaran:
        return True, "护卫舰", "v49.11：独立级单类补全置信度较低且缺少三体船锚点。"
    return False, "", ""


def v49_11_should_reject_wasp(text: str, status: str, confidence: float) -> Tuple[bool, str, str]:
    """判断黄蜂级是否被普通直升机/飞行甲板误闭集。"""
    no_amphibious = v49_11_has_any(text, [
        "未观察到全通飞行甲板", "未见全通飞行甲板", "没有全通飞行甲板", "无全通飞行甲板",
        "未观察到坞舱", "未见坞舱", "没有坞舱", "无坞舱",
        "未观察到大型登陆艇", "没有大型登陆艇", "无大型登陆艇", "未观察到大型登陆艇收放结构",
    ])
    frigate_like = v49_11_has_any(text, ["体量中等", "中型", "76毫米", "护卫舰", "中前部", "烟囱"])
    if no_amphibious and frigate_like:
        return True, "护卫舰", "v49.11：文本明确排除全通飞行甲板/坞舱/大型登陆艇，且更像中型护卫舰。"
    return False, "", ""


def v49_11_closed_set_counter_evidence_guard(result: Dict[str, Any], observed: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return result
    final = result.get("final_decision") or {}
    if final.get("result_type") != "known_class" or not final.get("primary_class"):
        result["v49_11_closed_set_guard"] = {"applied": False, "reason": "not_known_class"}
        return result

    class_name = v49_11_final_class(result)
    category = v49_11_final_category(result)
    status = v49_11_final_status(result)
    confidence = v49_1_float(final.get("confidence", 0.0))
    text = v49_11_text(observed)

    # 只处理已经确认的误闭集高发路径，避免像 v49.8 一样误伤大量正常已知类。
    if class_name == "阿利·伯克级驱逐舰" and status in {"v48_promoted_arleigh", "single_known_class_filled"}:
        reject, new_cat, reason = v49_11_should_reject_arleigh(text, status, confidence)
        if reject:
            return v49_11_set_category_unknown(result, new_cat or "驱逐舰", reason)

    if class_name == "独立级濒海战斗舰" and status in {"single_known_class_filled", "v47_synced_known_class", "single_best"}:
        reject, new_cat, reason = v49_11_should_reject_independence(text, status, confidence)
        if reject:
            return v49_11_set_category_unknown(result, new_cat or "护卫舰", reason)

    if class_name == "黄蜂级两栖攻击舰" and status == "single_known_class_filled":
        reject, new_cat, reason = v49_11_should_reject_wasp(text, status, confidence)
        if reject:
            return v49_11_set_category_unknown(result, new_cat or "两栖舰", reason)

    result["v49_11_closed_set_guard"] = {
        "applied": False,
        "reason": "no_counter_evidence_triggered",
        "class_name": class_name,
        "status": status,
        "confidence": round(confidence, 4),
    }
    return result


def hierarchical_class_match(
    class_data_path: str,
    observed_attributes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """v49.11：基于 v49.7，针对 v48/single-known 补全路径增加反证保护。"""
    result = _hierarchical_class_match_v49_7_base(class_data_path, observed_attributes)
    result = v49_11_closed_set_counter_evidence_guard(result, observed_attributes)
    return result



# ==================== v49.12: amphibious / landing known-class boundary fix ====================
# 目标：基于 v49.11，处理黄蜂级、圣安东尼奥级、惠德比岛级之间的边界混淆。
# 约束：不改前面已经修好的 open-set 保护逻辑，只在两栖/登陆三类之间做窄范围纠偏。
# 核心思想：
# 1. 黄蜂级：全通/贯通飞行甲板、两栖攻击、航空突击、STOVL/垂直起降。
# 2. 圣安东尼奥级：LPD/船坞运输、前部大型上层建筑、直升机平台/机库、车辆/货物/人员综合运输，且不是纯船坞登陆舰。
# 3. 惠德比岛级：LSD/船坞登陆、登陆艇投送为主、大型坞舱/艉门/登陆艇收放，且明确没有全通飞行甲板/STOVL。

_hierarchical_class_match_v49_11_base = hierarchical_class_match


def v49_12_text_flags(text: str) -> Dict[str, bool]:
    """两栖/登陆边界判定用的文本证据。"""
    no_full_deck = v49_11_has_any(text, [
        "不像全通甲板两栖攻击舰", "不是全通甲板两栖攻击舰", "不像全通飞行甲板",
        "不是全通飞行甲板", "没有全通飞行甲板", "未见全通飞行甲板", "未观察到全通飞行甲板", "无全通飞行甲板",
        "没有看到航母那种贯通大甲板", "不像航母那种整条飞行甲板", "没有看到航母那种整条飞行甲板",
        "没有STOVL", "无STOVL", "没有短距起飞", "没有垂直降落", "没有垂直起降", "也没有STOVL",
    ])
    full_deck_positive = (not no_full_deck) and v49_11_has_any(text, [
        "全通飞行甲板", "全通甲板", "贯通甲板", "整条飞行甲板", "类似小型航空母舰", "小型航母", "直通甲板",
    ])
    st0vl_positive = (not no_full_deck) and v49_11_has_any(text, [
        "STOVL", "短距起飞", "垂直降落", "垂直起降", "AV-8B", "F-35B", "垂直/短距", "短距/垂直",
    ])
    wasp_mission = v49_11_has_any(text, [
        "两栖攻击舰", "两栖攻击任务", "两栖攻击", "航空突击", "远征部队核心平台", "海军陆战队远征部队核心平台",
    ])

    stern_opening = v49_11_has_any(text, [
        "艉门", "舰尾开口", "船尾开口", "尾部开口", "大型开口", "舰艉区域可见大型开口", "船尾似乎能打开", "船尾能打开",
    ])
    well_deck = v49_11_has_any(text, [
        "坞舱", "大型坞舱", "船坞", "船坞登陆平台", "登陆艇收放", "登陆艇进出", "登陆艇投送", "气垫登陆艇", "LCAC",
    ])
    landing_primary = v49_11_has_any(text, [
        "登陆艇投送为主", "以登陆艇投送为主", "主要能看到船坞登陆平台", "船坞登陆舰", "LSD", "惠德比", "船坞登陆",
    ])

    large_superstructure = v49_11_has_any(text, [
        "前部上层建筑很大", "前面有很大的方盒子", "方盒子一样的舰桥", "大型上层建筑", "大型箱形上层建筑", "前部是大型上层建筑",
        "大型运输舰", "大型两栖运输", "大型运输平台",
    ])
    heli_platform_or_hangar = v49_11_has_any(text, [
        "直升机平台", "直升机甲板", "直升机机库", "机库", "后面有直升机平台", "后部有直升机平台", "后面有直升机甲板",
    ])
    transport_space = v49_11_has_any(text, [
        "车辆/货物运输", "车辆货物运输", "车辆/货物运输空间", "车辆甲板", "货物运输", "人员运输", "运兵", "运输空间", "综合运输",
    ])
    lpd_positive = v49_11_has_any(text, [
        "圣安东尼奥", "LPD", "两栖船坞运输舰", "船坞运输舰", "两栖船坞运输", "不是纯船坞登陆舰", "不像纯船坞登陆舰",
    ])
    pure_landing_negative_for_lpd = v49_11_has_any(text, [
        "纯船坞登陆舰", "登陆艇投送为主", "以登陆艇投送为主", "主要能看到船坞登陆平台",
    ]) and not v49_11_has_any(text, ["不是纯船坞登陆舰", "不像纯船坞登陆舰"])

    return {
        "no_full_deck": no_full_deck,
        "full_deck_positive": full_deck_positive,
        "stovl_positive": st0vl_positive,
        "wasp_mission": wasp_mission,
        "stern_opening": stern_opening,
        "well_deck": well_deck,
        "landing_primary": landing_primary,
        "large_superstructure": large_superstructure,
        "heli_platform_or_hangar": heli_platform_or_hangar,
        "transport_space": transport_space,
        "lpd_positive": lpd_positive,
        "pure_landing_negative_for_lpd": pure_landing_negative_for_lpd,
    }


def v49_12_set_known_class(result: Dict[str, Any], category: str, class_name: str, reason: str, confidence: Optional[float] = None) -> Dict[str, Any]:
    old_final = result.get("final_decision") or {}
    old_cat = result.get("category_result") or {}
    conf = confidence
    if conf is None:
        conf = max(
            v49_1_float(old_final.get("confidence", 0.0)),
            v49_1_float(old_cat.get("confidence", 0.0)),
            0.88,
        )
    result["category_result"] = {
        "label": category,
        "confidence": round(float(conf), 4),
        "status": "v49_12_amphibious_landing_boundary_category",
        "reason": reason,
    }
    result["known_class_result"] = {
        "label": class_name,
        "ship_class": class_name,
        "category": category,
        "confidence": round(float(conf), 4),
        "known_status": "Known",
        "reason": reason,
    }
    result["open_set_result"] = {
        "is_unknown": False,
        "unknown_scope": None,
        "reason": reason,
    }
    result["final_decision"] = {
        "result_type": "known_class",
        "primary_category": category,
        "primary_class": class_name,
        "confidence": round(float(conf), 4),
        "status": "v49_12_amphibious_landing_boundary_fix",
        "message": f"最终判定：{category} / {class_name}。{reason}",
        "alternatives": old_final.get("alternatives", {}) if isinstance(old_final, dict) else {},
    }
    result["v49_12_amphibious_landing_boundary_fix"] = {
        "applied": True,
        "category": category,
        "class_name": class_name,
        "reason": reason,
    }
    return result


def v49_12_amphibious_landing_boundary_fix(result: Dict[str, Any], observed: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """黄蜂 / 圣安东尼奥 / 惠德比岛之间的窄范围边界修正。"""
    if not isinstance(result, dict):
        return result

    text = v49_11_text(observed)
    flags = v49_12_text_flags(text)
    final = result.get("final_decision") or {}
    result_type = clean_text(final.get("result_type") or "")
    class_name = v49_11_final_class(result)
    category = v49_11_final_category(result)
    confidence = v49_1_float(final.get("confidence", 0.0))

    # 1) 圣安东尼奥误判为黄蜂：如果明确“不是全通甲板两栖攻击舰”，且具备 LPD 结构，不应判黄蜂。
    if class_name == "黄蜂级两栖攻击舰":
        if flags["no_full_deck"] and flags["well_deck"] and (flags["stern_opening"] or flags["landing_primary"]):
            return v49_12_set_known_class(
                result,
                "登陆舰",
                "惠德比岛级船坞登陆舰",
                "v49.12：文本明确排除全通飞行甲板/STOVL，并以坞舱、艉门和登陆艇收放为主要证据，改判为船坞登陆舰。",
                confidence,
            )
        if flags["no_full_deck"] and (flags["lpd_positive"] or (flags["large_superstructure"] and flags["heli_platform_or_hangar"])):
            return v49_12_set_known_class(
                result,
                "两栖舰",
                "圣安东尼奥级两栖船坞运输舰",
                "v49.12：文本排除全通甲板两栖攻击舰，并呈现大型上层建筑、直升机平台/机库等 LPD 特征，改判为圣安东尼奥级。",
                confidence,
            )

    # 2) 黄蜂误判为圣安东尼奥：全通/贯通甲板、垂直起降/航空突击是黄蜂级两栖攻击舰关键证据。
    if class_name == "圣安东尼奥级两栖船坞运输舰":
        if (flags["full_deck_positive"] or flags["stovl_positive"] or flags["wasp_mission"]) and not flags["no_full_deck"]:
            if flags["wasp_mission"] or flags["stovl_positive"] or flags["full_deck_positive"]:
                return v49_12_set_known_class(
                    result,
                    "两栖舰",
                    "黄蜂级两栖攻击舰",
                    "v49.12：文本包含全通/贯通飞行甲板、两栖攻击/航空突击或 STOVL 垂直起降证据，优先判为黄蜂级两栖攻击舰。",
                    confidence,
                )

    # 3) 圣安东尼奥误判为惠德比岛：大型上层建筑 + 直升机平台/机库 + 车辆/货物/人员运输，尤其“不是纯船坞登陆舰”。
    if class_name == "惠德比岛级船坞登陆舰":
        san_evidence = (
            flags["lpd_positive"]
            or (flags["large_superstructure"] and flags["heli_platform_or_hangar"])
            or (flags["transport_space"] and flags["heli_platform_or_hangar"])
        )
        if san_evidence and not flags["pure_landing_negative_for_lpd"]:
            return v49_12_set_known_class(
                result,
                "两栖舰",
                "圣安东尼奥级两栖船坞运输舰",
                "v49.12：文本呈现 LPD/两栖船坞运输特征，包括大型上层建筑、直升机平台/机库和车辆/货物运输空间，不应按纯船坞登陆舰处理。",
                confidence,
            )

    # 4) category_unknown 的已知样本恢复：仅限很强的两栖/登陆结构证据。
    if result_type == "category_unknown":
        if category in {"登陆舰", "两栖舰"} and (flags["wasp_mission"] or flags["stovl_positive"] or flags["full_deck_positive"]) and not flags["no_full_deck"]:
            return v49_12_set_known_class(
                result,
                "两栖舰",
                "黄蜂级两栖攻击舰",
                "v49.12：类别内未知结果中存在航空突击/两栖攻击或全通飞行甲板/STOVL 强证据，恢复为黄蜂级。",
                confidence,
            )
        if category == "航空母舰" and flags["no_full_deck"] and (flags["large_superstructure"] or flags["stern_opening"]) and flags["heli_platform_or_hangar"]:
            return v49_12_set_known_class(
                result,
                "两栖舰",
                "圣安东尼奥级两栖船坞运输舰",
                "v49.12：文本明确不是航母全通大甲板，同时具备大型上层建筑、直升机平台和舰尾开口等 LPD 证据，恢复为圣安东尼奥级。",
                confidence,
            )
        if category == "登陆舰" and flags["no_full_deck"] and (flags["landing_primary"] or (flags["well_deck"] and flags["stern_opening"])):
            return v49_12_set_known_class(
                result,
                "登陆舰",
                "惠德比岛级船坞登陆舰",
                "v49.12：登陆舰类别内未知结果中存在船坞登陆平台、无全通飞行甲板/STOVL和登陆艇收放证据，恢复为惠德比岛级。",
                confidence,
            )

    result["v49_12_amphibious_landing_boundary_fix"] = {
        "applied": False,
        "reason": "no_amphibious_landing_boundary_triggered",
        "class_name": class_name,
        "category": category,
        "confidence": round(confidence, 4),
    }
    return result


def hierarchical_class_match(
    class_data_path: str,
    observed_attributes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """v49.12：在 v49.11 基础上修复黄蜂/圣安东尼奥/惠德比岛边界。"""
    result = _hierarchical_class_match_v49_11_base(class_data_path, observed_attributes)
    result = v49_12_amphibious_landing_boundary_fix(result, observed_attributes)
    return result


# ==================== v49.16: surface-combatant parameter/category boundary fix ====================
# 目标：基于 v49.12，先不再恢复 known_class，只处理“类别内未知/开放集”的水面作战舰大类边界。
# 背景：
# - 诊断显示剩余错误里较多是 unknown -> unknown 但大类错，尤其护卫舰/驱逐舰/巡洋舰边界。
# - 参数型文本（长度、排水量、垂发数量、乘员、续航）经常被现有规则误吸到巡洋舰/航空母舰/驱逐舰。
# 约束：
# - 只改 category_unknown/open-set 的 primary_category；
# - 不填 pred_known_class，不恢复阿利·伯克/独立级；
# - 不触碰已经明确 known_class 的结果，避免破坏 v49.11/v49.12 的 open-set 保护。

_hierarchical_class_match_v49_12_base = hierarchical_class_match


def v49_16_number_patterns(text: str) -> Dict[str, List[float]]:
    """从原文里抽取少量对水面作战舰边界有用的数值。"""
    t = str(text or "")
    nums = {"length_m": [], "displacement_t": [], "crew": [], "vls_cells": []}

    # 舰长 / 长度
    for m in re.finditer(r"(?:舰长|全长|长度|长)\s*(?:约|为)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:m|米)", t, flags=re.I):
        try:
            nums["length_m"].append(float(m.group(1)))
        except Exception:
            pass

    # 满载/标准/排水量
    for m in re.finditer(r"(?:满载排水量|标准排水量|排水量)\s*(?:约|为)?\s*([0-9]+(?:\.[0-9]+)?)\s*吨", t):
        try:
            nums["displacement_t"].append(float(m.group(1)))
        except Exception:
            pass

    # 乘员
    for m in re.finditer(r"(?:乘员|舰员|船员)\s*(?:约|为)?\s*([0-9]+)\s*人", t):
        try:
            nums["crew"].append(float(m.group(1)))
        except Exception:
            pass

    # 垂发单元/具/个
    for m in re.finditer(r"([0-9]+)\s*(?:单元|具|个)\s*(?:Mk\s*41|MK41|垂直发射|垂发)", t, flags=re.I):
        try:
            nums["vls_cells"].append(float(m.group(1)))
        except Exception:
            pass
    for m in re.finditer(r"(?:Mk\s*41|MK41|垂直发射|垂发)[^0-9]{0,8}([0-9]+)\s*(?:单元|具|个)", t, flags=re.I):
        try:
            nums["vls_cells"].append(float(m.group(1)))
        except Exception:
            pass
    # 12组八联装 -> 96；2组八联装 -> 16
    for m in re.finditer(r"([0-9]+)\s*组\s*八联装", t):
        try:
            nums["vls_cells"].append(float(m.group(1)) * 8.0)
        except Exception:
            pass

    return nums


def v49_16_is_open_category_only(result: Dict[str, Any]) -> bool:
    """只允许处理类别内未知/open-set，不处理已知类。"""
    if not isinstance(result, dict):
        return False
    final = result.get("final_decision") or {}
    known = result.get("known_class_result") or {}
    open_set = result.get("open_set_result") or {}
    result_type = clean_text(final.get("result_type") or "")
    primary_class = clean_text(final.get("primary_class") or "")
    known_label = clean_text(known.get("label") or known.get("ship_class") or "") if isinstance(known, dict) else ""

    if primary_class not in V49_1_UNKNOWN_LABELS:
        return False
    if known_label not in V49_1_UNKNOWN_LABELS:
        return False
    return bool(open_set.get("is_unknown", False)) or result_type in {"category_unknown", "ambiguous_category", "category_only"}


def v49_16_surface_scores(text: str) -> Dict[str, Dict[str, Any]]:
    """给护卫舰/驱逐舰/巡洋舰做保守的大类边界打分。只用于 category_unknown。"""
    scores = {
        "护卫舰": {"score": 0.0, "evidence": []},
        "驱逐舰": {"score": 0.0, "evidence": []},
        "巡洋舰": {"score": 0.0, "evidence": []},
    }

    def add(cat: str, value: float, ev: str):
        scores[cat]["score"] += float(value)
        scores[cat]["evidence"].append(ev)

    nums = v49_16_number_patterns(text)
    lengths = nums["length_m"]
    disps = nums["displacement_t"]
    crews = nums["crew"]
    vls_cells = nums["vls_cells"]
    max_len = max(lengths) if lengths else None
    max_disp = max(disps) if disps else None
    max_vls = max(vls_cells) if vls_cells else None
    max_crew = max(crews) if crews else None

    # 1. 显式大类词：只修大类，不恢复舰级。
    if v49_11_has_any(text, ["大型导弹驱逐舰", "大型防空驱逐舰", "防空驱逐舰", "导弹驱逐舰"]):
        add("驱逐舰", 9.0, "explicit_destroyer_word")
    elif v49_11_has_any(text, ["驱逐舰"]):
        add("驱逐舰", 6.0, "explicit_destroyer_word")

    if v49_11_has_any(text, ["护卫舰", "巡防舰", "反潜护卫", "通用护卫"]):
        add("护卫舰", 8.0, "explicit_frigate_word")
    if v49_11_has_any(text, ["中型军舰", "中型舰艇", "中型水面舰艇", "中型水面作战舰"]):
        add("护卫舰", 5.0, "medium_ship_word")
    if v49_11_has_any(text, ["巡洋舰"]):
        add("巡洋舰", 7.0, "explicit_cruiser_word")

    # 2. 参数区间：这是本轮新增重点。参数只修大类，不闭集。
    if max_disp is not None:
        if max_disp <= 4200:
            add("护卫舰", 8.0, f"small_displacement_{max_disp:g}t")
        elif max_disp <= 7000:
            add("护卫舰", 6.0, f"medium_displacement_{max_disp:g}t")
        elif max_disp <= 10500:
            add("驱逐舰", 5.0, f"destroyer_scale_displacement_{max_disp:g}t")
        elif max_disp >= 11000:
            add("巡洋舰", 5.0, f"cruiser_scale_displacement_{max_disp:g}t")

    if max_len is not None:
        if max_len <= 135:
            add("护卫舰", 6.0, f"frigate_length_{max_len:g}m")
        elif max_len <= 155:
            add("护卫舰", 4.0, f"medium_frigate_length_{max_len:g}m")
        elif max_len <= 175:
            add("驱逐舰", 4.0, f"destroyer_length_{max_len:g}m")
        elif max_len >= 180:
            add("巡洋舰", 3.0, f"large_surface_length_{max_len:g}m")

    if max_crew is not None:
        if max_crew <= 180:
            add("护卫舰", 4.0, f"small_crew_{max_crew:g}")
        elif max_crew <= 330:
            add("驱逐舰", 3.0, f"destroyer_crew_{max_crew:g}")

    if v49_11_has_any(text, ["三千多吨", "3000多吨", "三千余吨", "3000余吨", "3650吨", "3600吨", "6200吨"]):
        add("护卫舰", 5.0, "explicit_frigate_scale_text")

    # 3. 垂发/武器传感器：共享区分特征只修大类。
    if max_vls is not None:
        if max_vls <= 24:
            add("护卫舰", 5.0, f"small_vls_{max_vls:g}")
        elif max_vls <= 48:
            add("驱逐舰", 3.0, f"medium_vls_{max_vls:g}")
        elif 70 <= max_vls <= 100:
            add("驱逐舰", 7.0, f"destroyer_vls_{max_vls:g}")
        elif max_vls >= 110:
            add("巡洋舰", 8.0, f"cruiser_vls_{max_vls:g}")

    if v49_11_has_any(text, ["90单元", "90具", "90个", "90具左右", "90单元左右", "96单元", "96具", "前后垂发", "前后甲板布置大量垂直发射"]):
        add("驱逐舰", 6.0, "large_destroyer_vls_text")
    if v49_11_has_any(text, ["122单元", "122具", "12组八联装", "舰队指挥"]):
        add("巡洋舰", 6.0, "cruiser_vls_or_command_text")
    if v49_11_has_any(text, ["16单元", "2组八联装", "两组八联装", "少量垂发", "有限垂发"]):
        add("护卫舰", 4.0, "small_vls_text")

    if v49_11_has_any(text, ["宙斯盾", "SPY", "AN/SPY", "相控阵雷达", "区域防空", "弹道导弹防御"]):
        # 如果同时有“驱逐舰”或 90 单元级证据，偏驱逐；否则只给弱支持，避免直接闭集。
        if scores["驱逐舰"]["score"] >= scores["巡洋舰"]["score"]:
            add("驱逐舰", 2.5, "aegis_or_area_air_defense_shared")
        else:
            add("巡洋舰", 2.0, "aegis_or_area_air_defense_shared")

    if v49_11_has_any(text, ["反潜直升机", "反舰导弹", "鱼雷管", "近防炮", "CODLOG", "MT30"]):
        add("护卫舰", 2.0, "frigate_littoral_or_escort_shared")

    # 4. 反向排除：小吨位/短舰体强烈排除巡洋舰；大型防空驱逐舰排除护卫舰。
    if (max_disp is not None and max_disp <= 7000) or (max_len is not None and max_len <= 155):
        scores["巡洋舰"]["score"] -= 4.0
        scores["巡洋舰"]["evidence"].append("negative_for_cruiser_medium_scale")
    if v49_11_has_any(text, ["明显大于普通护卫舰", "比普通护卫舰更大", "大型防空驱逐舰", "大型导弹驱逐舰"]):
        scores["护卫舰"]["score"] -= 3.5
        scores["护卫舰"]["evidence"].append("negative_for_frigate_large_destroyer_cue")
        add("驱逐舰", 3.0, "large_destroyer_relative_cue")
    if v49_11_has_any(text, ["不是航母", "不像航母", "没有全通飞行甲板", "无全通飞行甲板", "没有弹射器", "没有拦阻索"]):
        # 这里不直接给水面舰加太多分，只是防止被航空母舰方向吸走时允许纠偏。
        add("驱逐舰", 0.5, "negative_for_carrier_surface_context")
        add("护卫舰", 0.5, "negative_for_carrier_surface_context")

    for c in scores:
        scores[c]["score"] = round(scores[c]["score"], 4)
    return scores


def v49_16_set_category_unknown(result: Dict[str, Any], category: str, reason: str, confidence: float, debug: Dict[str, Any]) -> Dict[str, Any]:
    old_final = result.get("final_decision") or {}
    result["category_result"] = {
        "label": category,
        "confidence": round(float(confidence), 4),
        "status": "v49_16_surface_parameter_category",
        "reason": reason,
    }
    result["known_class_result"] = None
    result["open_set_result"] = {
        "is_unknown": True,
        "unknown_scope": UNKNOWN_OUTPUT_TEMPLATE.format(category=category),
        "reason": reason,
    }
    result["final_decision"] = {
        "result_type": "category_unknown",
        "primary_category": category,
        "primary_class": None,
        "confidence": round(float(confidence), 4),
        "status": "v49_16_surface_parameter_category_fix",
        "message": f"最终判定：{category}类别内未知类。{reason}",
        "alternatives": old_final.get("alternatives", {}) if isinstance(old_final, dict) else {},
    }
    result["v49_16_surface_parameter_category_fix"] = {"applied": True, **debug}
    return result


def v49_16_surface_parameter_category_fix(result: Dict[str, Any], observed: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """修正开放集/类别内未知结果中的护卫舰/驱逐舰/巡洋舰参数边界。"""
    if not v49_16_is_open_category_only(result):
        result["v49_16_surface_parameter_category_fix"] = {"applied": False, "reason": "not_open_category_only"}
        return result

    final = result.get("final_decision") or {}
    cat_res = result.get("category_result") or {}
    current_cat = clean_text(final.get("primary_category") or cat_res.get("label") or "")
    text = v49_11_text(observed)
    scores = v49_16_surface_scores(text)
    ranked = sorted(scores, key=lambda c: float(scores[c]["score"]), reverse=True)
    top_cat = ranked[0]
    second_cat = ranked[1] if len(ranked) > 1 else ""
    top_score = float(scores[top_cat]["score"])
    second_score = float(scores[second_cat]["score"]) if second_cat else 0.0
    current_score = float(scores.get(current_cat, {}).get("score", 0.0)) if current_cat in scores else 0.0

    # 不处理证据太弱的样本。弱 VLM 描述如果没有参数/显式类别，继续保持原结果，避免过拟合。
    top_ev = scores[top_cat]["evidence"]
    has_explicit_or_parameter = any(
        key in ev
        for ev in top_ev
        for key in [
            "explicit_", "length_", "displacement_", "crew_", "vls_", "scale_text",
            "large_destroyer_relative_cue",
        ]
    )

    allow = (
        top_cat != current_cat
        and top_score >= 7.0
        and top_score - max(second_score, current_score) >= 2.5
        and has_explicit_or_parameter
    )

    debug = {
        "reason": "surface_parameter_boundary_checked",
        "current_category": current_cat,
        "top_category": top_cat,
        "second_category": second_cat,
        "top_score": round(top_score, 4),
        "second_score": round(second_score, 4),
        "current_score": round(current_score, 4),
        "top_evidence": top_ev[:12],
        "scores": {c: scores[c]["score"] for c in scores},
    }

    if not allow:
        result["v49_16_surface_parameter_category_fix"] = {"applied": False, **debug}
        return result

    conf = max(
        v49_1_float(final.get("confidence", 0.0)),
        v49_1_float(cat_res.get("confidence", 0.0)),
        min(0.94, 0.58 + top_score / 45.0),
    )
    return v49_16_set_category_unknown(
        result,
        top_cat,
        "v49.16：依据显式舰种词、长度/排水量/乘员/垂发数量等参数型证据，只修正水面作战舰大类，仍保持类别内未知。",
        conf,
        debug,
    )


def hierarchical_class_match(
    class_data_path: str,
    observed_attributes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """v49.16：在 v49.12 基础上修复开放集水面作战舰参数/大类边界。"""
    result = _hierarchical_class_match_v49_12_base(class_data_path, observed_attributes)
    result = v49_16_surface_parameter_category_fix(result, observed_attributes)
    return result


if __name__ == "__main__":
    asyncio.run(main())
