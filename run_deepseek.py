# 1. 彻底修复 Windows 控制台编码问题
import sys
import os
import json

# 强制 Python 进入 UTF-8 模式
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LANG"] = "en_US.UTF-8"
os.environ["LC_ALL"] = "en_US.UTF-8"

# 2. 修补 httpx 的 header 编码函数，使其支持 UTF-8
import httpx._models
_original_normalize = httpx._models._normalize_header_value
def _utf8_normalize(value, encoding=None):
    if isinstance(value, bytes):
        return value
    # 直接使用 UTF-8 编码，而不是 ASCII
    return value.encode("utf-8")
httpx._models._normalize_header_value = _utf8_normalize

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import asyncio

# Windows 下使用更稳定的 Selector 事件循环
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# ==================== 本地 Embedding 模型 ====================
from sentence_transformers import SentenceTransformer
import numpy as np
from nano_graphrag._utils import wrap_embedding_func_with_attrs

# 加载中文优化的 Embedding 模型（首次运行会自动下载约 1.3GB）
EMBED_MODEL = SentenceTransformer("BAAI/bge-large-zh-v1.5")

# 获取模型的向量维度和最大 token 长度
_embedding_dim = EMBED_MODEL.get_sentence_embedding_dimension()   # 1024
_max_token_size = EMBED_MODEL.max_seq_length                     # 512

@wrap_embedding_func_with_attrs(embedding_dim=_embedding_dim, max_token_size=_max_token_size)
async def local_embedding(texts: list[str]) -> np.ndarray:
    """本地 Embedding 函数，使用 BAAI/bge-large-zh-v1.5 模型"""
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)


from openai import AsyncOpenAI
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash

from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件到环境变量

# --- 模型配置 ---
MODEL = "deepseek-ai/DeepSeek-V3"
BASE_URL = "https://api.siliconflow.cn/v1"
API_KEY = os.getenv("SILICONFLOW_API_KEY")  # 从环境变量读取

# --- 自定义大模型调用函数 ---
async def siliconflow_llm_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # 处理缓存
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,  # 👈 只加这一行！
        **kwargs
    )
    result = response.choices[0].message.content

    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": MODEL}})

    return result





#=================================打印实体和关系====================================
def print_entities_and_relations(working_dir):
    """从 GraphML 文件中读取并打印所有实体和关系"""
    import networkx as nx

    graphml_file = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")

    if not os.path.exists(graphml_file):
        print(f"GraphML 文件不存在: {graphml_file}")
        return

    G = nx.read_graphml(graphml_file)

    print("\n" + "=" * 60)
    print(f"知识图谱统计: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
    print("=" * 60)

    # 按实体类型分组打印节点
    from collections import defaultdict
    type_groups = defaultdict(list)
    for node_id, node_data in G.nodes(data=True):
        etype = node_data.get("entity_type", "UNKNOWN").strip('"')
        name = node_id.strip('"')
        desc = node_data.get("description", "").strip('"')
        type_groups[etype].append((name, desc))

    print("\n【实体列表（按类型分组）】")
    for etype, items in sorted(type_groups.items()):
        print(f"\n--- {etype} ({len(items)} 个) ---")
        for name, desc in items:
            if desc and desc != "无":
                print(f"  · {name} | 描述: {desc}")
            else:
                print(f"  · {name}")

    # 打印关系
    print("\n【关系列表（按类型分组）】")
    rel_type_groups = defaultdict(list)
    for src, tgt, edge_data in G.edges(data=True):
        rel_desc = edge_data.get("description", "").strip('"')
        # 从描述中提取关系类型（第一个词）
        rel_type = rel_desc.split("，")[0] if rel_desc else "未知关系"
        rel_type_groups[rel_type].append((src.strip('"'), tgt.strip('"')))

    for rel_type, pairs in sorted(rel_type_groups.items()):
        print(f"\n--- {rel_type} ({len(pairs)} 个) ---")
        for src, tgt in pairs:
            print(f"  · {src} → {tgt}")

    print("\n" + "=" * 60)









# ==================== 纯文本解析模块 ====================
async def direct_text_parse(user_text: str) -> str:
    """只根据用户输入文本提取属性，不做任何外部知识查询"""

    parse_prompt = f"""你是一个军舰文本解析专家。请从以下用户输入的文本中提取军舰属性。

重要规则：
1. 只能使用用户输入文本中明确提到的信息，绝对不要补充、推断或猜测任何文本中没有的信息。
2. 对于视觉属性，同时提供两个字段：
   - "original": 用户在文本中的原始描述（若未提及则为空字符串）。
   - "normalized": 根据以下规则进行标准化：
     * 如果用户的描述足够具体，可以唯一或高度对应某个标准术语，则输出该标准术语。
       例如：“船头下面有个圆圆的鼓包” → “球鼻首”；“屁股是方的” → “方形船尾”；“甲板不是直的，是斜的” → “斜角甲板”。
     * 如果用户提到了该部位，但明确表示看不清、不知道具体类型（例如“桅杆看不太清楚”），则输出 "不确定"。
     * 如果用户完全没有提到该部位，则输出 "未知"。
3. 标准术语参考：
   - 舰首：球鼻首、舰首锐削、直立舰首
   - 舰尾：方形船尾、舰尾收缩、圆形舰尾
   - 舰岛：描述应包含层数信息（如“三层窗户”对应“三层舰岛”）或位置信息（如“在右边”对应“位于右舷”），标准化为“X层舰岛”“舰岛位于右/左舷”
   - 甲板：斜角甲板、全通甲板、滑跃甲板
   - 桅杆：封闭式桅杆、桁格桅、多面体桅杆
4. 装备类属性（雷达、武器、飞机等）只提取用户明确提到的型号或特征；完全没有提到则设为空列表 []。
5. 非视觉属性（尺寸、排水量等）如果用户未提及，一律设为 "未知"。
6. 严格按照指定的 JSON 格式输出，不要有任何额外文字。

用户输入文本：
{user_text}

你必须严格按照以下 JSON Schema 输出，所有槽位都必须出现：

{{
  "observed_attributes": {{
    "visual": {{
      "Bow_Feature": {{
        "original": "用户原文中对舰首的描述，若未提及则为空字符串",
        "normalized": "球鼻首/舰首锐削/直立舰首/不确定/未知"
      }},
      "Stern_Feature": {{
        "original": "用户原文中对舰尾的描述，若未提及则为空字符串",
        "normalized": "方形船尾/舰尾收缩/圆形舰尾/不确定/未知"
      }},
      "Island_Feature": {{
        "original": "用户原文中对舰岛的描述，若未提及则为空字符串",
        "normalized": "三层舰岛位于右舷/二层舰岛位于左舷/不确定/未知"
      }},
      "Deck_Feature": {{
        "original": "用户原文中对甲板的描述，若未提及则为空字符串",
        "normalized": "斜角甲板/全通甲板/滑跃甲板/不确定/未知"
      }},
      "Mast_Feature": {{
        "original": "用户原文中对桅杆的描述，若未提及则为空字符串",
        "normalized": "封闭式桅杆/桁格桅/多面体桅杆/不确定/未知"
      }}
    }},
    "non_visual": {{
      "Length_Overall": "文本中提到的舰总长或 未知",
      "Beam": "文本中提到的舷宽或 未知",
      "Flight_Deck_Width": "文本中提到的飞行甲板宽度或 未知",
      "Draft": "文本中提到的吃水深度或 未知",
      "Standard_Displacement": "文本中提到的标准排水量或 未知",
      "Full_Load_Displacement": "文本中提到的满载排水量或 未知",
      "Speed": "文本中提到的航速或 未知",
      "Range": "文本中提到的续航力或 未知",
      "Crew": "文本中提到的舰员数量或 未知",
      "Aircraft_Capacity": "文本中提到的舰载机数量或 未知",
      "Aviation_Fuel": "文本中提到的舰载航油或 未知",
      "Power_Output": "文本中提到的推进功率或 未知",
      "Propulsion": "文本中提到的推进装置描述或 未知",
      "Flight_Deck_Area": "文本中提到的飞行甲板面积或 未知",
      "Island_Position": "文本中提到的舰岛位置或 未知"
    }},
    "equipment_mentioned": {{
      "Radar_System": ["文本中提到的雷达型号或特征，若未提及则为空列表"],
      "Countermeasure_System": ["文本中提到的对抗/电子战系统或特征，若未提及则为空列表"],
      "Combat_System": ["文本中提到的指挥作战系统或特征，若未提及则为空列表"],
      "Weapon_System": ["文本中提到的导弹/近防武器等，若未提及则为空列表"],
      "Shipboard_Gun": ["文本中提到的舰载火炮，若未提及则为空列表"],
      "Aircraft": ["文本中提到的舰载飞机型号或特征，若未提及则为空列表"],
      "Powerplant": ["文本中提到的动力装置，若未提及则为空列表"]
    }}
  }},
  "textual_summary": "只基于用户输入文本，生成一句简洁的中文总结"
}}

只输出 JSON，不要任何解释或 markdown 标记。"""

    # 直接调用硅基流动的 LLM，不走 GraphRAG
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": parse_prompt}],
        temperature=0.0   # 确定性输出
    )
    return response.choices[0].message.content











# ==================== 图谱匹配（含置信度分层） ====================
async def match_candidates(graph_func, observed_attrs_json: str) -> str:
    """
    用观察到的属性在知识图谱中匹配候选舰船。
    强制要求 LLM 对每一个非空/非“未知”的槽位进行逐一比对。
    """
    # 解析观察属性
    observed = json.loads(observed_attrs_json)
    visual = observed.get("visual", {})
    non_visual = observed.get("non_visual", {})
    equipment = observed.get("equipment_mentioned", {})

    # 构建“已知条件”列表，只包含有值的属性
    known_conditions = []

    # 1. 视觉属性
    for key, value in visual.items():
        if value and value != "未知" and value != "不确定":
            known_conditions.append(f"  - {key}: {value}")

    # 2. 非视觉属性
    for key, value in non_visual.items():
        if value and value != "未知":
            known_conditions.append(f"  - {key}: {value}")

    # 3. 装备属性
    for sys_type, items in equipment.items():
        if items:
            known_conditions.append(f"  - {sys_type}: {', '.join(items)}")

    # 如果没有提取到任何有效条件，直接返回提示
    if not known_conditions:
        return json.dumps({
            "matched_candidates": [],
            "match_level": "no_match",
            "suggestion": "未能从输入文本中提取到任何有效属性，无法进行匹配。"
        }, ensure_ascii=False, indent=2)

    # 构造强制比对 Prompt
    known_text = "\n".join(known_conditions)

    query = f"""请严格按照以下规则进行舰船型号匹配。

## 已知条件（用户输入中提取的属性）
这些是唯一可用的匹配依据，必须全部使用：
{known_text}

## 匹配规则
1. **唯一性特征优先**：如果已知条件中包含具体的装备型号（如“勃朗宁 M2 重机枪”、“MK-91 NSSM”等），这属于高区分度特征。只有真正装备了该型号的舰船才能作为候选，未装备该型号的舰船直接排除。
2. **通用特征综合评分**：对于舰型、排水量范围、吃水深度等通用特征，允许模糊匹配（如“约10万吨”可匹配 9.7万-10.4万吨）。
3. **逐一比对**：必须对每一个已知条件在知识图谱中进行验证，并在 match_points 中列明每个条件的匹配情况。
4. **排除规则**：如果候选舰船的某个属性与已知条件明确冲突，必须在 differences 中说明，并显著降低其置信度。

## 知识图谱数据
{{context_data}}

请输出符合之前 JSON Schema 的匹配结果。
"""

    raw_result = await graph_func.aquery(query, param=QueryParam(mode="local"))

    try:
        # 清理可能的 markdown 标记
        if raw_result.startswith("```"):
            lines = raw_result.split("\n")
            lines = lines[1:] if lines[0].startswith("```") else lines
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw_result = "\n".join(lines)
        result = json.loads(raw_result)
    except json.JSONDecodeError:
        return json.dumps({"error": "匹配结果解析失败", "raw_output": raw_result}, ensure_ascii=False, indent=2)

    candidates = result.get("matched_candidates", [])
    if not candidates:
        result["match_level"] = "no_match"
        result["suggestion"] = "知识图谱中未找到满足条件的实体，该舰可能不在当前数据库中。建议以图像识别为主。"
        with open("pending_entities.json", "a", encoding="utf-8") as f:
            f.write(observed_attrs_json + "\n")
    else:
        max_conf = max(c.get("confidence", 0) for c in candidates)
        if max_conf > 0.7:
            result["match_level"] = "high_confidence"
            result["suggestion"] = "文本信息较充分，匹配结果可信度较高。"
        elif max_conf > 0.3:
            result["match_level"] = "low_confidence"
            result["suggestion"] = "文本信息不足，匹配结果仅供参考。强烈建议结合图像进行联合判断。"
        else:
            result["match_level"] = "very_low_confidence"
            result["suggestion"] = "文本信息严重不足，无法有效区分候选型号。请以图像识别结果为主要依据。"

    return json.dumps(result, ensure_ascii=False, indent=2)












# --- 主程序 ---
async def main():
    graph_func = GraphRAG(
        working_dir="./ship_index",
        best_model_func=siliconflow_llm_complete,
        cheap_model_func=siliconflow_llm_complete,
        best_model_id=MODEL,
        cheap_model_id=MODEL,
        embedding_func=local_embedding,   # 启用本地 Embedding
    )

    with open("./naval_data.txt", "r", encoding='utf-8') as f:
        await graph_func.ainsert(f.read())




    # ========== 打印实体和关系（用于调试） ==========
    print_entities_and_relations(graph_func.working_dir)

    # ========== 测试用例 ==========
    user_text = "一艘很大的航母，装备勃朗宁 M2 重机枪"
    print("=" * 60)
    print("【步骤1】纯文本解析结果：")
    parse_result = await direct_text_parse(user_text)
    print(parse_result)

    # 提取标准化后的属性
    try:
        parsed = json.loads(parse_result)
        observed_for_match = {
            "visual": {k: v.get("normalized", "未知") for k, v in parsed["observed_attributes"]["visual"].items()},
            "non_visual": parsed["observed_attributes"]["non_visual"],
            "equipment_mentioned": parsed["observed_attributes"]["equipment_mentioned"]
        }
        observed_json = json.dumps(observed_for_match, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"解析JSON失败: {e}")
        return

    print("\n" + "=" * 60)
    print("【步骤2】图谱匹配结果：")
    match_result = await match_candidates(graph_func, observed_json)
    print(match_result)



if __name__ == "__main__":
    asyncio.run(main())
