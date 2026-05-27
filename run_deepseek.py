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















# ==================== 受控实体词典与通用实体名规范化 ====================
import re
import unicodedata

KNOWN_ENTITIES = {
    "SHIP_CLASS": [
        "尼米兹级",
    ],

    "SHIP_INSTANCE": [
        "CVN-68 尼米兹号",
        "CVN-69 艾森豪威尔号",
        "CVN-70 卡尔文森号",
        "CVN-71 西奥多·罗斯福号",
        "CVN-72 亚伯拉罕·林肯号",
        "CVN-73 乔治·华盛顿号",
        "CVN-74 约翰·C·斯坦尼斯号",
        "CVN-75 哈里·S·杜鲁门号",
        "CVN-76 罗纳德·里根号",
        "CVN-77 乔治·H·W·布什号",
    ],

    # 固定视觉属性节点
    "BOW": ["船首"],
    "STERN": ["船尾"],
    "DECK": ["甲板"],
    "ISLAND": ["舰岛"],
    "MAST": ["桅杆"],

    # 固定文本属性节点
    "LENGTH_OVERALL": ["舰总长"],
    "BEAM": ["舷宽"],
    "FLIGHT_DECK_WIDTH": ["飞行甲板宽"],
    "DRAFT": ["吃水深度"],
    "STANDARD_DISPLACEMENT": ["标准排水量"],
    "FULL_LOAD_DISPLACEMENT": ["满载排水量"],
    "SPEED": ["航速"],
    "RANGE": ["续航力"],
    "CREW": ["舰员编制"],
    "AIRCRAFT_CAPACITY": ["舰载机数量"],
    "POWER_OUTPUT": ["推进功率"],
    "PROPULSION": ["推进装置"],
    "FLIGHT_DECK_AREA": ["飞行甲板面积"],
    "ISLAND_POSITION": ["舰岛位置"],
    "HOMEPORT": ["母港"],

    # 辅助固定节点
    "SHIPYARD": ["建造船厂"],
    "SERVICE_STATUS": ["服役状态"],

    # 装备类
    "POWERPLANT": [
        "A4W 压水核反应堆",
        "A4W/A1G 压水核反应堆",
        "蒸汽涡轮发动机",
        "四轴双主舵",
        "四轴四桨",
        "四轴五桨",
        "四桨四轴双舵",
        "汽轮发电机",
        "应急柴油发电机",
        "备用柴油机",
    ],

    "CATAPULT": [
        "弹射器",
    ],

    "ARRESTING_GEAR": [
        "拦阻索",
        "拦阻网",
    ],

    "RADAR_SYSTEM": [
        "AN/SPS-48C/E",
        "AN/SPS-48E",
        "AN/SPS-49(V)1",
        "AN/SPS-49(V)5",
        "AN/SPS-43A",
        "AN/SPS-67",
        "AN/SPS-67V",
        "AN/SPS-67V-1",
        "AN/SPQ-9A",
        "AN/SPQ-9B",
        "AN/SPN-46",
        "AN/SPN-43C",
        "AN/SPN-41",
        "AN/SPN-44",
        "Mk 91 NSSM",
        "Mk 95",
        "MK91-1",
        "MK-73",
        "SPS-64(V)9",
        "LN-66",
        "URN-25",
        "MK23 TAS",
    ],

    "RADAR_FUNCTION": [
        "对空搜索",
        "对海搜索",
        "火控",
        "空中管制",
        "目标截获",
        "导航",
        "测速",
    ],

    "COUNTERMEASURE_SYSTEM": [
        "AN/SLQ-32(V)4",
        "SLY-2",
        "AN/WLR-1H",
        "Mk 36 SRBOC",
        "AN/SLQ-25",
        "SLQ-25A",
        "SLQ-29",
        "SLQ-36",
    ],

    "COUNTERMEASURE_FUNCTION": [
        "电子战",
        "电子侦察",
        "诱饵发射",
        "拖曳鱼雷诱饵",
        "电子干扰",
    ],

    "COMBAT_SYSTEM": [
        "ACDS",
        "ACDS Block 0/1",
        "ACDS Block 1",
        "NTDS",
        "SSDS Mk 2",
        "MK-23 TAS",
    ],

    "COMBAT_FUNCTION": [
        "战斗指挥",
        "战术数据",
        "舰艇自卫",
        "目标搜获",
        "信息指挥",
    ],

    "COMMUNICATION_SYSTEM": [
        "SRR-1",
        "WSC-3",
        "WSC-6",
        "USC-38",
        "SSQ-82",
        "SQQ-1",
        "JOTS",
        "POST",
        "CVIC",
        "TESS UMM-1(V)1",
        "JMCIS",
        "SSQ-1A",
        "全光纤数字化通信系统",
        "IT-21",
    ],

    "COMMUNICATION_FUNCTION": [
        "卫星通信",
        "战术环境支援",
        "航母情报",
        "指挥信息系统",
        "联合战术系统",
    ],

    "DATA_LINK": [
        "LINK-4A",
        "LINK-11",
        "LINK-14",
        "LINK-16",
    ],

    "WEAPON_SYSTEM": [
        "Mk 25",
        "Mk 29",
        "Mk 31",
        "Mk 49",
        "Mk 57 Mod 3",
        "RIM-7",
        "RIM-7M",
        "RIM-116",
        "Mk 15",
        "LOCUST",
        "三联装324毫米鱼雷发射管",
    ],

    "WEAPON_FUNCTION": [
        "短程防空",
        "近防系统",
        "导弹发射装置",
        "激光武器",
        "鱼雷发射装置",
    ],

    "SHIPBOARD_GUN": [
        "Mk 38",
        "勃朗宁 M2",
    ],

    "SHIPBOARD_GUN_FUNCTION": [
        "遥控机炮",
        "重机枪",
    ],

    "AIRCRAFT": [
        "F/A-18E/F",
        "F/A-18C/D",
        "F/A-18A/B/C/D",
        "F/A-18A/C/E",
        "F/A-18F",
        "F/A-18",
        "F-14",
        "F-14D",
        "F-14A/B/D",
        "F-35C",
        "E-2C",
        "E-2D",
        "E-2",
        "EA-6B",
        "EA-18G",
        "A-6E",
        "S-3A/B",
        "S-3A",
        "S-3B",
        "ES-3A",
        "SH-3G/H",
        "SH-3G",
        "SH-3H",
        "SH-60F",
        "HH-60H",
        "MH-60R",
        "MH-60R/S",
        "SH-60",
        "UH-60",
        "C-2",
        "C-2A",
    ],

    "AIRCRAFT_FUNCTION": [
        "战斗攻击机",
        "电子战飞机",
        "预警机",
        "反潜机",
        "侦察机",
        "运输机",
        "直升机",
    ],

    "ARMOR_PROTECTION": [
        "双层舰壳",
        "X 形吸能支撑结构",
        "HY-80 高强度钢",
        "水密隔舱壁",
        "防火隔壁",
        "水密隔舱",
        "纵向防雷舱壁",
        "凯夫拉装甲",
        "先进灭火系统",
        "高强度合金钢",
        "多层隔离防护结构",
        "隐身吸波材料",
        "高弹性钢",
        "泡沫消防装置",
        "双层船体",
        "X形构件",
        "多层隔舱防护",
        "箱型防御结构",
    ],
}


ENTITY_TYPE_ALIASES = {
    "Ship_Class": "SHIP_CLASS",
    "Ship_Instance": "SHIP_INSTANCE",
    "Bow": "BOW",
    "Stern": "STERN",
    "Deck": "DECK",
    "Island": "ISLAND",
    "Mast": "MAST",
    "Radar_System": "RADAR_SYSTEM",
    "Countermeasure_System": "COUNTERMEASURE_SYSTEM",
    "Combat_System": "COMBAT_SYSTEM",
    "Communication_System": "COMMUNICATION_SYSTEM",
    "Data_Link": "DATA_LINK",
    "Weapon_System": "WEAPON_SYSTEM",
    "Shipboard_Gun": "SHIPBOARD_GUN",
    "Aircraft": "AIRCRAFT",
    "Powerplant": "POWERPLANT",
    "Catapult": "CATAPULT",
    "Arresting_Gear": "ARRESTING_GEAR",
    "Radar_Function": "RADAR_FUNCTION",
    "Countermeasure_Function": "COUNTERMEASURE_FUNCTION",
    "Combat_Function": "COMBAT_FUNCTION",
    "Communication_Function": "COMMUNICATION_FUNCTION",
    "Weapon_Function": "WEAPON_FUNCTION",
    "Aircraft_Function": "AIRCRAFT_FUNCTION",
    "Shipboard_Gun_Function": "SHIPBOARD_GUN_FUNCTION",
    "Armor_Protection": "ARMOR_PROTECTION",
    "Shipyard": "SHIPYARD",
    "Service_Status": "SERVICE_STATUS",
    "Length_Overall": "LENGTH_OVERALL",
    "Beam": "BEAM",
    "Flight_Deck_Width": "FLIGHT_DECK_WIDTH",
    "Draft": "DRAFT",
    "Standard_Displacement": "STANDARD_DISPLACEMENT",
    "Full_Load_Displacement": "FULL_LOAD_DISPLACEMENT",
    "Speed": "SPEED",
    "Range": "RANGE",
    "Crew": "CREW",
    "Aircraft_Capacity": "AIRCRAFT_CAPACITY",
    "Power_Output": "POWER_OUTPUT",
    "Propulsion": "PROPULSION",
    "Flight_Deck_Area": "FLIGHT_DECK_AREA",
    "Island_Position": "ISLAND_POSITION",
    "Homeport": "HOMEPORT",
    "Configuration": "CONFIGURATION",
}


SHIP_DEPENDENT_TYPES = {
    # 视觉固定槽位
    "BOW", "STERN", "DECK", "ISLAND", "MAST",

    # 纯文本属性固定槽位
    "LENGTH_OVERALL", "BEAM", "FLIGHT_DECK_WIDTH", "DRAFT",
    "STANDARD_DISPLACEMENT", "FULL_LOAD_DISPLACEMENT", "SPEED",
    "RANGE", "CREW", "AIRCRAFT_CAPACITY", "POWER_OUTPUT",
    "PROPULSION", "FLIGHT_DECK_AREA", "ISLAND_POSITION", "HOMEPORT",

    # 固定设备名，但具体型号随舰变化
    "CATAPULT", "ARRESTING_GEAR",

    # 辅助固定槽位
    "SHIPYARD", "SERVICE_STATUS",
}


def kg_clean_name(x) -> str:
    return str(x or "").strip().strip('"')


def kg_entity_type(entity_type: str) -> str:
    entity_type = str(entity_type or "UNKNOWN").strip().strip('"')
    return ENTITY_TYPE_ALIASES.get(entity_type, entity_type).upper()


def kg_lookup_key(name: str) -> str:
    """
    只用于匹配，不作为最终实体名写入图谱。
    目的：让 ACDS BLOCK 0/1、ACDS Block 0/1、ACDS block 0/1 映射到同一个 key。
    """
    s = kg_clean_name(name)

    if not s:
        return ""

    s = unicodedata.normalize("NFKC", s)

    # 统一不同连字符
    s = (
        s.replace("-", "-")
         .replace("–", "-")
         .replace("—", "-")
         .replace("－", "-")
    )

    # 去掉空格、下划线、引号；大小写统一仅用于比较
    s = re.sub(r'[\s_"“”\'`]+', "", s)
    s = s.casefold()

    return s


def build_known_entity_index():
    by_type = {}
    all_types = {}

    for etype, names in KNOWN_ENTITIES.items():
        etype = kg_entity_type(etype)
        by_type[etype] = {}

        for name in names:
            key = kg_lookup_key(name)
            if key:
                by_type[etype][key] = name

                # 全局索引用于兜底；同名冲突时保留第一次出现
                all_types.setdefault(key, name)

    return by_type, all_types


KNOWN_ENTITY_INDEX_BY_TYPE, KNOWN_ENTITY_INDEX_ALL = build_known_entity_index()


def canonicalize_entity_name(raw_name: str, entity_type: str = None, extra_names=None, allow_unknown: bool = False) -> str:
    """
    词典驱动的实体名标准化。

    返回：
    - 标准实体名
    - "" 表示无法映射到词典，不应写入主图
    """
    raw_name = kg_clean_name(raw_name)

    if not raw_name or raw_name in {"未知", "无"}:
        return ""

    etype = kg_entity_type(entity_type)

    # Configuration 是动态实体，来自 naval_data 的 [CONFIGURATION] 标题
    if extra_names:
        extra_index = {kg_lookup_key(x): x for x in extra_names}
        key = kg_lookup_key(raw_name)

        if key in extra_index:
            return extra_index[key]

    # 优先按实体类型匹配
    key = kg_lookup_key(raw_name)

    if etype in KNOWN_ENTITY_INDEX_BY_TYPE and key in KNOWN_ENTITY_INDEX_BY_TYPE[etype]:
        return KNOWN_ENTITY_INDEX_BY_TYPE[etype][key]

    # 再全局兜底
    if key in KNOWN_ENTITY_INDEX_ALL:
        return KNOWN_ENTITY_INDEX_ALL[key]

    # 视觉具体描述不能作为实体名，统一回固定节点
    visual_alias = {
        "球鼻艏": "船首",
        "球鼻首": "船首",
        "舰首锐削": "船首",
        "直立舰首": "船首",
        "方形船尾": "船尾",
        "舰尾收缩": "船尾",
        "圆形舰尾": "船尾",
        "斜角甲板": "甲板",
        "全通甲板": "甲板",
        "滑跃甲板": "甲板",
        "斜角甲板，直角甲板": "甲板",
        "舰岛位于右舷": "舰岛",
        "舰岛位于右舷中部": "舰岛",
        "舰岛位于右舷，靠近舰艉": "舰岛",
        "复合桅杆（与舰桥融为一体的封闭式桅杆）": "桅杆",
        "塔状桅杆，与舰岛整合": "桅杆",
        "细长高大的柱状综合桅杆": "桅杆",
    }

    if raw_name in visual_alias:
        return visual_alias[raw_name]

    # IT-21 的文本变体统一
    it21_aliases = {
        "IT21",
        "IT-21",
        "IT-21",
        "IT-21 非保密型局域网系统",
        "IT-21 非保密型局域网系统",
        'IT21"21世纪信息技术"系统',
        "21世纪信息技术系统",
    }

    if raw_name in it21_aliases:
        return "IT-21"

    # S-3B 说明性文本统一
    if raw_name in {
        "S-3B（只执行水面侦查任务）",
        "S-3B(只执行水面侦查任务)",
    }:
        return "S-3B"

    if raw_name == "ES-3Av":
        return "ES-3A"

    return raw_name if allow_unknown else ""


def split_entity_and_description(raw_name: str, entity_type: str = None, extra_names=None):
    """
    将 弹射器(C-13-1)、拦阻索(型号未知)、A-6E(退役) 拆成：
    - 标准实体名
    - 当前舰船上的说明信息

    注意：
    如果完整字符串本身能匹配词典实体，例如 AN/SPS-49(V)5，则不拆。
    """
    raw_name = kg_clean_name(raw_name)

    if not raw_name or raw_name in {"未知", "无"}:
        return "", "无"

    # 先尝试完整匹配，避免误拆 AN/SPS-49(V)5 这种型号
    full_match = canonicalize_entity_name(
        raw_name,
        entity_type=entity_type,
        extra_names=extra_names,
        allow_unknown=False
    )

    if full_match:
        return full_match, "无"

    # 再拆括号
    m = re.match(r"^(.+?)[\(（](.+?)[\)）]$", raw_name)

    if m:
        base_name = kg_clean_name(m.group(1))
        desc = kg_clean_name(m.group(2))

        std_name = canonicalize_entity_name(
            base_name,
            entity_type=entity_type,
            extra_names=extra_names,
            allow_unknown=False
        )

        return std_name, desc

    std_name = canonicalize_entity_name(
        raw_name,
        entity_type=entity_type,
        extra_names=extra_names,
        allow_unknown=False
    )

    return std_name, "无"























# --- LLM 调用限流器：强制串行请求，避免 SiliconFlow TPM 429 ---
_LLM_SEMAPHORE = None

def get_llm_semaphore():
    global _LLM_SEMAPHORE
    if _LLM_SEMAPHORE is None:
        _LLM_SEMAPHORE = asyncio.Semaphore(1)
    return _LLM_SEMAPHORE




# --- 自定义大模型调用函数 ---
async def siliconflow_llm_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    **kwargs
) -> str:
    """
    SiliconFlow LLM 调用函数。

    作用：
    1. 支持 nano-graphrag 的 hashing_kv 缓存
    2. 429 / TPM limit reached 时等待重试
    3. APITimeoutError / ReadTimeout 时等待重试
    4. 用 semaphore 强制串行请求，避免并发打爆 TPM
    """
    import random
    import httpx
    from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIConnectionError

    if history_messages is None:
        history_messages = []

    # nano-graphrag 会传入 hashing_kv，这个不能传给 OpenAI API
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)

    messages = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    messages.extend(history_messages)

    messages.append({
        "role": "user",
        "content": prompt
    })

    # 命中缓存则直接返回，减少 API 调用
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    else:
        args_hash = None

    # 保证抽取稳定
    kwargs.setdefault("temperature", 0.0)

    # 避免把外部传入的重复参数搞冲突
    kwargs.pop("model", None)
    kwargs.pop("messages", None)

    # 关闭 SDK 自带短间隔重试，改用我们自己的长等待重试
    # read timeout 拉长，避免长输出时直接超时
    client = AsyncOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        max_retries=0,
        timeout=httpx.Timeout(
            connect=30.0,
            read=600.0,
            write=120.0,
            pool=120.0
        ),
    )

    max_retries = 10

    for retry_idx in range(max_retries):
        try:
            async with get_llm_semaphore():
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    **kwargs
                )

            result = response.choices[0].message.content

            if hashing_kv is not None and args_hash is not None:
                await hashing_kv.upsert({
                    args_hash: {
                        "return": result,
                        "model": MODEL
                    }
                })

            return result

        except RateLimitError:
            wait_seconds = min(300, 30 * (retry_idx + 1)) + random.uniform(0, 8)
            print(
                f"[LLM限流] TPM limit reached，"
                f"第 {retry_idx + 1}/{max_retries} 次重试，"
                f"等待 {wait_seconds:.1f} 秒..."
            )
            await asyncio.sleep(wait_seconds)

        except (APITimeoutError, APIConnectionError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            wait_seconds = min(300, 20 * (retry_idx + 1)) + random.uniform(0, 8)
            print(
                f"[LLM超时/网络波动] {type(e).__name__}，"
                f"第 {retry_idx + 1}/{max_retries} 次重试，"
                f"等待 {wait_seconds:.1f} 秒..."
            )
            await asyncio.sleep(wait_seconds)

        except Exception as e:
            error_text = str(e)

            if (
                "429" in error_text
                or "TPM limit" in error_text
                or "rate limiting" in error_text.lower()
                or "rate limit" in error_text.lower()
                or "Too Many Requests" in error_text
            ):
                wait_seconds = min(300, 30 * (retry_idx + 1)) + random.uniform(0, 8)
                print(
                    f"[LLM限流] 捕获到限流异常，"
                    f"第 {retry_idx + 1}/{max_retries} 次重试，"
                    f"等待 {wait_seconds:.1f} 秒..."
                )
                await asyncio.sleep(wait_seconds)

            elif (
                "timeout" in error_text.lower()
                or "timed out" in error_text.lower()
                or "ReadTimeout" in error_text
            ):
                wait_seconds = min(300, 20 * (retry_idx + 1)) + random.uniform(0, 8)
                print(
                    f"[LLM超时] 捕获到超时异常，"
                    f"第 {retry_idx + 1}/{max_retries} 次重试，"
                    f"等待 {wait_seconds:.1f} 秒..."
                )
                await asyncio.sleep(wait_seconds)

            else:
                raise e

    raise RuntimeError("LLM 调用多次重试后仍然失败：限流或超时")





















# ==================== GraphRAG 对象构建函数 ====================
def build_graph_rag(working_dir="./ship_index"):
    """
    统一创建 GraphRAG 对象。

    注意：
    1. best_model_max_async / cheap_model_max_async 控制内部并发
    2. entity_extract_max_gleaning=0 关闭继续追问抽取，减少 LLM 请求
    3. chunk_token_size 降低单次请求 token 压力
    """
    return GraphRAG(
        working_dir=working_dir,

        best_model_func=siliconflow_llm_complete,
        cheap_model_func=siliconflow_llm_complete,

        best_model_id=MODEL,
        cheap_model_id=MODEL,

        embedding_func=local_embedding,

        # 降低 nano-graphrag 内部 LLM 并发
        best_model_max_async=1,
        cheap_model_max_async=1,

        # 关闭 continue_prompt / gleaning，避免额外多轮抽取请求
        entity_extract_max_gleaning=0,

        # 降低每个 chunk 的 token 压力
        chunk_token_size=700,
        chunk_overlap_token_size=80,
    )


# ==================== 图谱关系方向规范化 ====================
def normalize_graph_directions(working_dir: str, naval_data_path: str = "./naval_data.txt"):
    """
    读取 nano_graphrag 生成的 graph_chunk_entity_relation.graphml，
    根据实体类型 + 关系类型强制修正边方向，并删除明显不符合 schema 的边。

    注意：
    这个函数修改的是真实 GraphML 文件，不是只修改打印内容。
    """
    import os
    import re
    import shutil
    import networkx as nx

    graphml_file = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")

    if not os.path.exists(graphml_file):
        print(f"[方向修正] GraphML 文件不存在: {graphml_file}")
        return

    backup_file = graphml_file + ".bak_before_direction_fix"
    if not os.path.exists(backup_file):
        shutil.copy2(graphml_file, backup_file)

    G = nx.read_graphml(graphml_file)

    def clean_name(x):
        return str(x).strip().strip('"')

    def clean_type(x):
        return str(x or "UNKNOWN").strip().strip('"').upper()

    # 从 naval_data 中读取合法 Ship_Instance 和 Configuration
    valid_ship_instances = set()
    valid_configurations = set()

    if naval_data_path and os.path.exists(naval_data_path):
        with open(naval_data_path, "r", encoding="utf-8") as f:
            naval_text = f.read()

        for m in re.finditer(r"^Ship_Instance:\s*(.+?)\s*$", naval_text, flags=re.M):
            valid_ship_instances.add(clean_name(m.group(1)))

        config_section_matches = re.finditer(
            r"\[CONFIGURATION\](.*?)(?=\[/SHIP\])",
            naval_text,
            flags=re.S
        )
        for cm in config_section_matches:
            config_text = cm.group(1)
            for hm in re.finditer(r"^(CVN-\d+\s+[^\n:：]+)[：:]\s*$", config_text, flags=re.M):
                valid_configurations.add(clean_name(hm.group(1)))

    REL_SCHEMA = {
        "INSTANCE_OF": ({"SHIP_CLASS"}, {"SHIP_INSTANCE"}),

        "BOW_OF": ({"BOW"}, {"SHIP_INSTANCE"}),
        "STERN_OF": ({"STERN"}, {"SHIP_INSTANCE"}),
        "DECK_OF": ({"DECK"}, {"SHIP_INSTANCE"}),
        "ISLAND_OF": ({"ISLAND"}, {"SHIP_INSTANCE"}),
        "MAST_OF": ({"MAST"}, {"SHIP_INSTANCE"}),

        "LENGTH_OVERALL_OF": ({"LENGTH_OVERALL"}, {"SHIP_INSTANCE"}),
        "BEAM_OF": ({"BEAM"}, {"SHIP_INSTANCE"}),
        "FLIGHT_DECK_WIDTH_OF": ({"FLIGHT_DECK_WIDTH"}, {"SHIP_INSTANCE"}),
        "DRAFT_OF": ({"DRAFT"}, {"SHIP_INSTANCE"}),
        "STANDARD_DISPLACEMENT_OF": ({"STANDARD_DISPLACEMENT"}, {"SHIP_INSTANCE"}),
        "FULL_LOAD_DISPLACEMENT_OF": ({"FULL_LOAD_DISPLACEMENT"}, {"SHIP_INSTANCE"}),
        "SPEED_OF": ({"SPEED"}, {"SHIP_INSTANCE"}),
        "RANGE_OF": ({"RANGE"}, {"SHIP_INSTANCE"}),
        "CREW_OF": ({"CREW"}, {"SHIP_INSTANCE"}),
        "AIRCRAFT_CAPACITY_OF": ({"AIRCRAFT_CAPACITY"}, {"SHIP_INSTANCE"}),
        "POWER_OUTPUT_OF": ({"POWER_OUTPUT"}, {"SHIP_INSTANCE"}),
        "PROPULSION_OF": ({"PROPULSION"}, {"SHIP_INSTANCE"}),
        "FLIGHT_DECK_AREA_OF": ({"FLIGHT_DECK_AREA"}, {"SHIP_INSTANCE"}),
        "ISLAND_POSITION_OF": ({"ISLAND_POSITION"}, {"SHIP_INSTANCE"}),
        "HOMEPORT_OF": ({"HOMEPORT"}, {"SHIP_INSTANCE"}),

        "BUILT": ({"SHIPYARD"}, {"SHIP_INSTANCE"}),
        "SERVICE_STATUS_OF": ({"SERVICE_STATUS"}, {"SHIP_INSTANCE"}),
        "ARMOR_PROTECTION_OF": ({"ARMOR_PROTECTION"}, {"SHIP_INSTANCE"}),

        "EQUIPPED_WITH": ({"SHIP_INSTANCE"}, {"CONFIGURATION"}),

        "RADAR_OF": ({"RADAR_SYSTEM"}, {"CONFIGURATION"}),
        "WEAPON_OF": ({"WEAPON_SYSTEM"}, {"CONFIGURATION"}),
        "COUNTERMEASURE_OF": ({"COUNTERMEASURE_SYSTEM"}, {"CONFIGURATION"}),
        "COMBAT_SYSTEM_OF": ({"COMBAT_SYSTEM"}, {"CONFIGURATION"}),
        "COMMUNICATION_OF": ({"COMMUNICATION_SYSTEM"}, {"CONFIGURATION"}),
        "DATA_LINK_OF": ({"DATA_LINK"}, {"CONFIGURATION"}),
        "GUN_OF": ({"SHIPBOARD_GUN"}, {"CONFIGURATION"}),
        "AIRCRAFT_OF": ({"AIRCRAFT"}, {"CONFIGURATION"}),
        "POWERPLANT_OF": ({"POWERPLANT"}, {"CONFIGURATION"}),
        "CATAPULT_OF": ({"CATAPULT"}, {"CONFIGURATION"}),
        "ARRESTING_GEAR_OF": ({"ARRESTING_GEAR"}, {"CONFIGURATION"}),
        "ARMOR_OF": ({"ARMOR_PROTECTION"}, {"CONFIGURATION"}),

        "HAS_RADAR_FUNCTION": ({"RADAR_SYSTEM"}, {"RADAR_FUNCTION"}),
        "HAS_COUNTERMEASURE_FUNCTION": ({"COUNTERMEASURE_SYSTEM"}, {"COUNTERMEASURE_FUNCTION"}),
        "HAS_COMBAT_FUNCTION": ({"COMBAT_SYSTEM"}, {"COMBAT_FUNCTION"}),
        "HAS_COMMUNICATION_FUNCTION": ({"COMMUNICATION_SYSTEM"}, {"COMMUNICATION_FUNCTION"}),
        "HAS_WEAPON_FUNCTION": ({"WEAPON_SYSTEM"}, {"WEAPON_FUNCTION"}),
        "HAS_AIRCRAFT_FUNCTION": ({"AIRCRAFT"}, {"AIRCRAFT_FUNCTION"}),
        "HAS_SHIPBOARD_GUN_FUNCTION": ({"SHIPBOARD_GUN"}, {"SHIPBOARD_GUN_FUNCTION"}),
    }

    known_rel_types = set(REL_SCHEMA.keys())

    def merge_text(old_text, new_text):
        old_text = str(old_text or "")
        new_text = str(new_text or "")

        if not new_text:
            return old_text
        if not old_text:
            return new_text

        parts = old_text.split("<SEP>")
        if new_text in parts:
            return old_text

        return old_text + "<SEP>" + new_text

    node_attrs = {}
    node_name_map = {}

    for n, data in G.nodes(data=True):
        raw_name = clean_name(n)
        etype = clean_type(data.get("entity_type", "UNKNOWN"))

        extra = valid_configurations if etype == "CONFIGURATION" else None

        canonical_name = canonicalize_entity_name(
            raw_name,
            entity_type=etype,
            extra_names=extra,
            allow_unknown=False
        )

        if not canonical_name:
            continue

        node_name_map[raw_name] = canonical_name

        fixed_data = dict(data)
        fixed_data["entity_type"] = etype

        # 对固定属性节点，不再拼接每艘舰不同的 description
        if etype in SHIP_DEPENDENT_TYPES:
            fixed_data["description"] = "固定属性节点"

        if canonical_name not in node_attrs:
            node_attrs[canonical_name] = fixed_data
        else:
            node_attrs[canonical_name]["description"] = merge_text(
                node_attrs[canonical_name].get("description", ""),
                fixed_data.get("description", "")
            )
            node_attrs[canonical_name]["source_id"] = merge_text(
                node_attrs[canonical_name].get("source_id", ""),
                fixed_data.get("source_id", "")
            )

    def get_node_type(name):
        name = clean_name(name)
        return clean_type(node_attrs.get(name, {}).get("entity_type", "UNKNOWN"))

    def is_valid_node(name, data):
        name = clean_name(name)
        etype = clean_type(data.get("entity_type", "UNKNOWN"))

        if not name:
            return False

        if etype == "UNKNOWN":
            return False

        if "<(" in name or "ENTITY" in name or "<" in name or ">" in name:
            return False

        if etype == "SHIP_INSTANCE" and valid_ship_instances:
            return name in valid_ship_instances

        if etype == "CONFIGURATION" and valid_configurations:
            return name in valid_configurations

        # 你当前设计中 纯文本实体名 固定
        fixed_name_by_type = {
            "BOW": {"船首"},
            "STERN": {"船尾"},
            "DECK": {"甲板"},
            "ISLAND": {"舰岛"},
            "MAST": {"桅杆"},
            "LENGTH_OVERALL": {"舰总长"},
            "BEAM": {"舷宽"},
            "FLIGHT_DECK_WIDTH": {"飞行甲板宽"},
            "DRAFT": {"吃水深度"},
            "STANDARD_DISPLACEMENT": {"标准排水量"},
            "FULL_LOAD_DISPLACEMENT": {"满载排水量"},
            "SPEED": {"航速"},
            "RANGE": {"续航力"},
            "CREW": {"舰员编制"},
            "AIRCRAFT_CAPACITY": {"舰载机数量"},
            "POWER_OUTPUT": {"推进功率"},
            "PROPULSION": {"推进装置"},
            "FLIGHT_DECK_AREA": {"飞行甲板面积"},
            "ISLAND_POSITION": {"舰岛位置"},
            "HOMEPORT": {"母港"},
            "CATAPULT": {"弹射器"},
            "ARRESTING_GEAR": {"拦阻索", "拦阻网"},
            "SHIPYARD": {"建造船厂"},
            "SERVICE_STATUS": {"服役状态"},
        }

        if etype in fixed_name_by_type:
            return name in fixed_name_by_type[etype]

        return True

        return True

    def extract_relation_type(edge_data):
        candidate = str(edge_data.get("relation_type", "")).strip().strip('"')

        if candidate:
            for part in candidate.split("<SEP>"):
                part = part.strip().upper()
                if part in known_rel_types:
                    return part

        desc = str(edge_data.get("description", "")).strip().strip('"')
        desc_upper = desc.upper()

        for rel in sorted(known_rel_types, key=len, reverse=True):
            if desc_upper.startswith(rel):
                return rel

        first = re.split(r"[，,；;\n<]", desc_upper)[0].strip()
        if first in known_rel_types:
            return first

        return ""

    def add_or_merge_edge(DG, src, tgt, edge_data, rel_type):
        src = clean_name(src)
        tgt = clean_name(tgt)

        if src not in DG.nodes or tgt not in DG.nodes:
            return

        new_data = {str(k): str(v) for k, v in dict(edge_data).items()}
        new_data["relation_type"] = rel_type

        if DG.has_edge(src, tgt):
            old_data = DG[src][tgt]

            old_desc = old_data.get("description", "")
            new_desc = new_data.get("description", "")
            if new_desc and new_desc not in old_desc:
                old_data["description"] = old_desc + "<SEP>" + new_desc if old_desc else new_desc

            old_src = old_data.get("source_id", "")
            new_src = new_data.get("source_id", "")
            if new_src and new_src not in old_src:
                old_data["source_id"] = old_src + "<SEP>" + new_src if old_src else new_src

            old_rel = old_data.get("relation_type", "")
            if rel_type and rel_type not in old_rel:
                old_data["relation_type"] = old_rel + "<SEP>" + rel_type if old_rel else rel_type
        else:
            DG.add_edge(src, tgt, **new_data)

    DG = nx.DiGraph()

    for name, data in node_attrs.items():
        if is_valid_node(name, data):
            fixed_data = dict(data)
            fixed_data["entity_type"] = clean_type(fixed_data.get("entity_type", "UNKNOWN"))
            DG.add_node(clean_name(name), **fixed_data)

    kept = 0
    fixed = 0
    dropped = 0

    if G.is_multigraph():
        edge_iter = ((u, v, data) for u, v, _key, data in G.edges(keys=True, data=True))
    else:
        edge_iter = G.edges(data=True)

    for src, tgt, edge_data in edge_iter:
        raw_src = clean_name(src)
        raw_tgt = clean_name(tgt)

        src = node_name_map.get(raw_src)
        tgt = node_name_map.get(raw_tgt)

        if not src or not tgt:
            dropped += 1
            continue

        if src not in DG.nodes or tgt not in DG.nodes:
            dropped += 1
            continue

        rel_type = extract_relation_type(edge_data)
        if rel_type not in REL_SCHEMA:
            dropped += 1
            continue

        expected_src_types, expected_tgt_types = REL_SCHEMA[rel_type]

        src_type = get_node_type(src)
        tgt_type = get_node_type(tgt)

        if src_type in expected_src_types and tgt_type in expected_tgt_types:
            add_or_merge_edge(DG, src, tgt, edge_data, rel_type)
            kept += 1

        elif tgt_type in expected_src_types and src_type in expected_tgt_types:
            add_or_merge_edge(DG, tgt, src, edge_data, rel_type)
            fixed += 1

        else:
            dropped += 1

    nx.write_graphml(DG, graphml_file)

    print(f"[方向修正完成] 保留 {kept} 条，反转 {fixed} 条，删除 {dropped} 条")
    print(f"[方向修正完成] 已写回真实 GraphML: {graphml_file}")
    print(f"[方向修正完成] 原始备份: {backup_file}")


# ==================== 根据 naval_data 强制补全确定性关系 ====================
def repair_graph_from_naval_data(working_dir: str, naval_data_path: str = "./naval_data.txt"):
    """
    从 naval_data.txt 的标准结构中直接补全确定性关系。
    这个函数会修改真实 GraphML 文件，不是只修改打印输出。
    """
    import os
    import re
    import shutil
    import networkx as nx

    graphml_file = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")

    if not os.path.exists(graphml_file):
        print(f"[规则构图] GraphML 不存在，将从 naval_data 直接创建新图: {graphml_file}")
        os.makedirs(working_dir, exist_ok=True)
        G = nx.DiGraph()
        backup_file = None
    else:
        backup_file = graphml_file + ".bak_before_repair"
        if not os.path.exists(backup_file):
            shutil.copy2(graphml_file, backup_file)

        G = nx.read_graphml(graphml_file)

    if not os.path.exists(naval_data_path):
        print(f"[补边] naval_data 文件不存在: {naval_data_path}")
        return


    with open(naval_data_path, "r", encoding="utf-8") as f:
        text = f.read()

    def clean_name(s):
        return kg_clean_name(s)

    def canonical_entity_type(entity_type: str) -> str:
        return kg_entity_type(entity_type)

    # 提前收集动态 Configuration 名称
    valid_configurations = set()

    config_section_matches = re.finditer(
        r"\[CONFIGURATION\](.*?)(?=\[/SHIP\])",
        text,
        flags=re.S
    )

    for cm in config_section_matches:
        config_text_for_names = cm.group(1)
        for hm in re.finditer(r"^(CVN-\d+\s+[^\n:：]+)[：:]\s*$", config_text_for_names, flags=re.M):
            valid_configurations.add(clean_name(hm.group(1)))

    def normalize_equipment_name(name, entity_type=None):
        entity_name, _ = split_entity_and_description(
            name,
            entity_type=entity_type,
            extra_names=valid_configurations
        )
        return entity_name

    def append_desc(old_desc, new_desc):
        old_desc = str(old_desc or "")
        new_desc = str(new_desc or "")

        if not new_desc:
            return old_desc

        if new_desc in old_desc:
            return old_desc

        if not old_desc:
            return new_desc

        return old_desc + "<SEP>" + new_desc

    def add_node_if_missing(name, entity_type, description="无"):
        entity_type = canonical_entity_type(entity_type)

        extra = valid_configurations if entity_type == "CONFIGURATION" else None

        name = canonicalize_entity_name(
            name,
            entity_type=entity_type,
            extra_names=extra,
            allow_unknown=False
        )

        if not name or name == "未知" or name == "无":
            return

        if entity_type in SHIP_DEPENDENT_TYPES:
            node_desc = "固定属性节点"
        else:
            node_desc = description

        if name not in G:
            G.add_node(
                name,
                entity_type=entity_type,
                description=node_desc,
                source_id="naval_data_repair"
            )
        else:
            old_type = str(G.nodes[name].get("entity_type", "")).strip().strip('"')

            if not old_type or old_type.upper() == "UNKNOWN":
                G.nodes[name]["entity_type"] = entity_type
            else:
                G.nodes[name]["entity_type"] = kg_entity_type(old_type)

            if entity_type in SHIP_DEPENDENT_TYPES:
                G.nodes[name]["description"] = "固定属性节点"
            else:
                G.nodes[name]["description"] = append_desc(
                    G.nodes[name].get("description", ""),
                    node_desc
                )

    def add_edge_if_missing(src, tgt, relation_type, description=""):
        src = clean_name(src)
        tgt = clean_name(tgt)

        if not src or not tgt or src == "未知" or tgt == "未知":
            return

        desc = description or f"{relation_type}，由 naval_data 结构化数据补全"

        if G.has_edge(src, tgt):
            G[src][tgt]["description"] = append_desc(G[src][tgt].get("description", ""), desc)
            G[src][tgt]["relation_type"] = append_desc(G[src][tgt].get("relation_type", ""), relation_type)
            return

        G.add_edge(
            src,
            tgt,
            description=desc,
            relation_type=relation_type,
            weight=1.0,
            source_id="naval_data_repair"
        )

    def get_section(block, section_name):
        pattern = rf"\[{re.escape(section_name)}\](.*?)(?=\n\[[A-Z_]+\]|\n\[/SHIP\]|\Z)"
        m = re.search(pattern, block, flags=re.S)
        return m.group(1).strip() if m else ""

    def parse_key_value_lines(section_text):
        result = {}

        for line in section_text.splitlines():
            line = clean_name(line)

            if not line or line.startswith("#"):
                continue

            if ":" in line:
                k, v = line.split(":", 1)
            elif "：" in line:
                k, v = line.split("：", 1)
            else:
                continue

            result[clean_name(k)] = clean_name(v)

        return result

    def split_items(value):
        value = clean_name(value)

        if not value or value in {"未知", "无"}:
            return []

        value = value.replace("，", ",")
        parts = [clean_name(x) for x in value.split(",")]
        return [x for x in parts if x and x not in {"未知", "无"}]

    visual_map = {
        "Bow": ("船首", "Bow", "BOW_OF"),
        "Stern": ("船尾", "Stern", "STERN_OF"),
        "Deck": ("甲板", "Deck", "DECK_OF"),
        "Island": ("舰岛", "Island", "ISLAND_OF"),
        "Mast": ("桅杆", "Mast", "MAST_OF"),
    }

    text_attr_map = {
        "Length_Overall": ("舰总长", "Length_Overall", "LENGTH_OVERALL_OF"),
        "Beam": ("舷宽", "Beam", "BEAM_OF"),
        "Flight_Deck_Width": ("飞行甲板宽", "Flight_Deck_Width", "FLIGHT_DECK_WIDTH_OF"),
        "Draft": ("吃水深度", "Draft", "DRAFT_OF"),
        "Standard_Displacement": ("标准排水量", "Standard_Displacement", "STANDARD_DISPLACEMENT_OF"),
        "Full_Load_Displacement": ("满载排水量", "Full_Load_Displacement", "FULL_LOAD_DISPLACEMENT_OF"),
        "Speed": ("航速", "Speed", "SPEED_OF"),
        "Range": ("续航力", "Range", "RANGE_OF"),
        "Crew": ("舰员编制", "Crew", "CREW_OF"),
        "Aircraft_Capacity": ("舰载机数量", "Aircraft_Capacity", "AIRCRAFT_CAPACITY_OF"),
        "Power_Output": ("推进功率", "Power_Output", "POWER_OUTPUT_OF"),
        "Propulsion": ("推进装置", "Propulsion", "PROPULSION_OF"),
        "Flight_Deck_Area": ("飞行甲板面积", "Flight_Deck_Area", "FLIGHT_DECK_AREA_OF"),
        "Island_Position": ("舰岛位置", "Island_Position", "ISLAND_POSITION_OF"),
        "Homeport": ("母港", "Homeport", "HOMEPORT_OF"),
    }

    config_type_map = {
        "雷达套件": ("RADAR_OF", "Radar_System"),
        "武器套件": ("WEAPON_OF", "Weapon_System"),
        "舰载火炮套件": ("GUN_OF", "Shipboard_Gun"),
        "电子战套件": ("COUNTERMEASURE_OF", "Countermeasure_System"),
        "作战系统套件": ("COMBAT_SYSTEM_OF", "Combat_System"),
        "通信套件": ("COMMUNICATION_OF", "Communication_System"),
        "数据链套件": ("DATA_LINK_OF", "Data_Link"),
        "舰载机联队": ("AIRCRAFT_OF", "Aircraft"),
        "动力套件": ("POWERPLANT_OF", "Powerplant"),
        "弹射器套件": ("CATAPULT_OF", "Catapult"),
        "拦阻装置套件": ("ARRESTING_GEAR_OF", "Arresting_Gear"),
        "装甲防护套件": ("ARMOR_OF", "Armor_Protection"),
    }

    function_section_map = {
        "RADAR_FUNCTION": ("HAS_RADAR_FUNCTION", "Radar_System", "Radar_Function"),
        "COUNTERMEASURE_FUNCTION": ("HAS_COUNTERMEASURE_FUNCTION", "Countermeasure_System", "Countermeasure_Function"),
        "COMBAT_FUNCTION": ("HAS_COMBAT_FUNCTION", "Combat_System", "Combat_Function"),
        "COMMUNICATION_FUNCTION": ("HAS_COMMUNICATION_FUNCTION", "Communication_System", "Communication_Function"),
        "WEAPON_FUNCTION": ("HAS_WEAPON_FUNCTION", "Weapon_System", "Weapon_Function"),
        "AIRCRAFT_FUNCTION": ("HAS_AIRCRAFT_FUNCTION", "Aircraft", "Aircraft_Function"),
        "SHIPBOARD_GUN_FUNCTION": ("HAS_SHIPBOARD_GUN_FUNCTION", "Shipboard_Gun", "Shipboard_Gun_Function"),
    }

    ship_blocks = re.findall(r"【([^】]+)】\s*\[SHIP\](.*?)\[/SHIP\]", text, flags=re.S)

    added_edges = 0

    for title, block in ship_blocks:
        ship_match = re.search(r"^Ship_Instance:\s*(.+?)\s*$", block, flags=re.M)
        class_match = re.search(r"^Ship_Class:\s*(.+?)\s*$", block, flags=re.M)

        if not ship_match:
            continue

        ship_instance = clean_name(ship_match.group(1))
        ship_class = clean_name(class_match.group(1)) if class_match else ""

        add_node_if_missing(ship_instance, "Ship_Instance")

        if ship_class:
            add_node_if_missing(ship_class, "Ship_Class")
            add_edge_if_missing(
                ship_class,
                ship_instance,
                "INSTANCE_OF",
                f"INSTANCE_OF，{ship_instance} 属于 {ship_class}"
            )
            added_edges += 1

        visual_kv = parse_key_value_lines(get_section(block, "VISUAL_FEATURES"))
        for key, (fixed_node, entity_type, rel_type) in visual_map.items():
            value = visual_kv.get(key, "无")
            add_node_if_missing(fixed_node, entity_type, value)
            add_edge_if_missing(
                fixed_node,
                ship_instance,
                rel_type,
                f"{rel_type}，{fixed_node} 属于 {ship_instance}，属性值：{value}"
            )
            added_edges += 1

        text_kv = parse_key_value_lines(get_section(block, "TEXT_ATTRIBUTES"))
        for key, (fixed_node, entity_type, rel_type) in text_attr_map.items():
            value = text_kv.get(key, "无")
            add_node_if_missing(fixed_node, entity_type, value)
            add_edge_if_missing(
                fixed_node,
                ship_instance,
                rel_type,
                f"{rel_type}，{fixed_node} 属于 {ship_instance}，属性值：{value}"
            )
            added_edges += 1

        shipyard_text = get_section(block, "SHIPYARD")
        shipyard_value = "、".join([clean_name(x) for x in shipyard_text.splitlines() if clean_name(x)])

        if shipyard_value:
            add_node_if_missing("建造船厂", "Shipyard", shipyard_value)
            add_edge_if_missing(
                "建造船厂",
                ship_instance,
                "BUILT",
                f"BUILT，{ship_instance} 建造船厂：{shipyard_value}"
            )
            added_edges += 1

        service_text = get_section(block, "SERVICE_STATUS")
        service_value = "、".join([clean_name(x) for x in service_text.splitlines() if clean_name(x)])

        if service_value:
            add_node_if_missing("服役状态", "Service_Status", service_value)
            add_edge_if_missing(
                "服役状态",
                ship_instance,
                "SERVICE_STATUS_OF",
                f"SERVICE_STATUS_OF，{ship_instance} 服役状态：{service_value}"
            )
            added_edges += 1

        armor_text = get_section(block, "ARMOR_PROTECTION")
        for line in armor_text.splitlines():
            armor = clean_name(line)

            if not armor or armor == "未知" or armor.startswith("#"):
                continue

            add_node_if_missing(armor, "Armor_Protection")
            add_edge_if_missing(
                armor,
                ship_instance,
                "ARMOR_PROTECTION_OF",
                f"ARMOR_PROTECTION_OF，{armor} 属于 {ship_instance}"
            )
            added_edges += 1

        config_text = get_section(block, "CONFIGURATION")

        if config_text:
            config_headers = list(re.finditer(r"^(CVN-\d+\s+[^\n:：]+)[：:]\s*$", config_text, flags=re.M))

            for i, header in enumerate(config_headers):
                config_name = clean_name(header.group(1))

                start = header.end()
                end = config_headers[i + 1].start() if i + 1 < len(config_headers) else len(config_text)
                items_text = config_text[start:end]

                relation_type = None
                equipment_entity_type = None

                for suffix, pair in config_type_map.items():
                    if config_name.endswith(suffix):
                        relation_type, equipment_entity_type = pair
                        break

                if not relation_type:
                    continue

                add_node_if_missing(config_name, "Configuration")

                add_edge_if_missing(
                    ship_instance,
                    config_name,
                    "EQUIPPED_WITH",
                    f"EQUIPPED_WITH，{ship_instance} 装备 {config_name}"
                )
                added_edges += 1

                for line in items_text.splitlines():
                    raw_item = clean_name(line)

                    if not raw_item or raw_item.startswith("#") or raw_item in {"未知", "无"}:
                        continue

                    item, item_desc = split_entity_and_description(
                        raw_item,
                        entity_type=equipment_entity_type,
                        extra_names=valid_configurations
                    )

                    if not item or item in {"未知", "无"}:
                        continue

                    add_node_if_missing(item, equipment_entity_type)

                    edge_desc = f"{relation_type}，{item} 属于 {config_name}"

                    if item_desc and item_desc != "无":
                        edge_desc += f"，说明：{item_desc}"

                    add_edge_if_missing(
                        item,
                        config_name,
                        relation_type,
                        edge_desc
                    )
                    added_edges += 1

        for section_name, (rel_type, equipment_type, function_type) in function_section_map.items():
            func_text = get_section(block, section_name)
            func_kv = parse_key_value_lines(func_text)

            for function_name, equipment_list_text in func_kv.items():
                function_name = clean_name(function_name)

                if not function_name or function_name in {"未知", "无"}:
                    continue

                add_node_if_missing(function_name, function_type)

                for equip in split_items(equipment_list_text):
                    equip, equip_desc = split_entity_and_description(
                        equip,
                        entity_type=equipment_type,
                        extra_names=valid_configurations
                    )

                    if not equip or equip in {"未知", "无"}:
                        continue

                    add_node_if_missing(equip, equipment_type)

                    edge_desc = f"{rel_type}，{equip} 具备功能 {function_name}"

                    if equip_desc and equip_desc != "无":
                        edge_desc += f"，说明：{equip_desc}"

                    add_edge_if_missing(
                        equip,
                        function_name,
                        rel_type,
                        edge_desc
                    )
                    added_edges += 1

    nx.write_graphml(G, graphml_file)

    print(f"[补边完成] 根据 naval_data 尝试补全/修正 {added_edges} 条确定性关系")
    print(f"[补边完成] 已写回真实 GraphML: {graphml_file}")
    if backup_file:
        print(f"[补边完成] 原始备份: {backup_file}")
    else:
        print("[规则构图完成] 本次为从 0 创建图谱，无原始备份")


# ==================== 图谱修复结果检查 ====================
def sanity_check_graph(working_dir: str):
    """
    检查几个关键关系是否已经写入真实 GraphML。
    """
    import os
    import networkx as nx

    graphml_file = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")

    if not os.path.exists(graphml_file):
        print(f"[检查] GraphML 文件不存在: {graphml_file}")
        return

    G = nx.read_graphml(graphml_file)

    checks = [
        ("勃朗宁 M2", "CVN-68 舰载火炮套件"),
        ("CVN-68 舰载火炮套件", "勃朗宁 M2"),
        ("推进功率", "CVN-68 尼米兹号"),
        ("CVN-68 尼米兹号", "推进功率"),
        ("船首", "CVN-68 尼米兹号"),
        ("CVN-68 尼米兹号", "船首"),
    ]

    print("\n" + "=" * 60)
    print("【图谱修复检查】")
    for src, tgt in checks:
        print(f"{src} → {tgt}: {G.has_edge(src, tgt)}")
    print("=" * 60)



























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
        rel_type = str(edge_data.get("relation_type", "")).strip('"')
        rel_desc = str(edge_data.get("description", "")).strip('"')

        # 优先使用后处理函数写入的 relation_type
        if rel_type:
            rel_type = rel_type.split("<SEP>")[0].strip()
        else:
            # 兼容旧图：从 description 中提取关系类型
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
















# ==================== GraphML 精确匹配函数 ====================
def normalize_observed_equipment_name(item: str, entity_type: str) -> str:
    """
    将用户输入解析出的装备名标准化为图谱实体名。
    例如：
    勃朗宁 M2 重机枪 -> 勃朗宁 M2
    """
    item = kg_clean_name(item)

    if not item or item in {"未知", "无"}:
        return ""

    # 先直接走词典标准化
    std = canonicalize_entity_name(
        item,
        entity_type=entity_type,
        allow_unknown=False
    )

    if std:
        return std

    # 处理常见口语/功能后缀
    alias_rules = {
        "勃朗宁 M2 重机枪": "勃朗宁 M2",
        "M2 重机枪": "勃朗宁 M2",
        "M2机枪": "勃朗宁 M2",
        "勃朗宁M2": "勃朗宁 M2",
        "Mk38": "Mk 38",
        "Mk 38 机炮": "Mk 38",
        "MK38": "Mk 38",
    }

    if item in alias_rules:
        return alias_rules[item]

    # 再尝试去掉部分中文描述词
    cleaned = (
        item.replace("重机枪", "")
            .replace("机枪", "")
            .replace("舰炮", "")
            .replace("火炮", "")
            .replace("系统", "")
            .strip()
    )

    std = canonicalize_entity_name(
        cleaned,
        entity_type=entity_type,
        allow_unknown=False
    )

    if std:
        return std

    return ""


def get_edge_relation_type(edge_data: dict) -> str:
    """
    读取边的 relation_type。
    """
    rel = str(edge_data.get("relation_type", "")).strip().strip('"')

    if rel:
        return rel.split("<SEP>")[0].strip()

    desc = str(edge_data.get("description", "")).strip().strip('"')
    if "，" in desc:
        return desc.split("，")[0].strip()

    if "," in desc:
        return desc.split(",")[0].strip()

    return ""


def get_ship_from_configuration(G, config_name: str):
    """
    根据 Configuration 节点反查属于哪艘舰。
    路径：
    Ship_Instance → Configuration
    relation_type = EQUIPPED_WITH
    """
    for src, tgt, data in G.in_edges(config_name, data=True):
        rel = get_edge_relation_type(data)
        src_type = str(G.nodes[src].get("entity_type", "")).upper()

        if rel == "EQUIPPED_WITH" and src_type == "SHIP_INSTANCE":
            return src

    return ""


def match_candidates_precise(working_dir: str, observed_attrs_json: str) -> str:
    """
    直接读取 GraphML 做精确匹配。
    不再依赖 graph_func.aquery，不再让 LLM 判断候选舰船。
    """
    import os
    import json
    import networkx as nx
    from collections import defaultdict

    graphml_file = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")

    if not os.path.exists(graphml_file):
        return json.dumps({
            "matched_candidates": [],
            "match_level": "no_match",
            "suggestion": f"GraphML 文件不存在: {graphml_file}"
        }, ensure_ascii=False, indent=2)

    G = nx.read_graphml(graphml_file)
    observed = json.loads(observed_attrs_json)

    equipment = observed.get("equipment_mentioned", {})
    visual = observed.get("visual", {})
    non_visual = observed.get("non_visual", {})

    equipment_schema = {
        "Radar_System": ("RADAR_SYSTEM", "RADAR_OF"),
        "Countermeasure_System": ("COUNTERMEASURE_SYSTEM", "COUNTERMEASURE_OF"),
        "Combat_System": ("COMBAT_SYSTEM", "COMBAT_SYSTEM_OF"),
        "Weapon_System": ("WEAPON_SYSTEM", "WEAPON_OF"),
        "Shipboard_Gun": ("SHIPBOARD_GUN", "GUN_OF"),
        "Aircraft": ("AIRCRAFT", "AIRCRAFT_OF"),
        "Powerplant": ("POWERPLANT", "POWERPLANT_OF"),
    }

    visual_schema = {
        "Bow_Feature": ("船首", "BOW_OF"),
        "Stern_Feature": ("船尾", "STERN_OF"),
        "Island_Feature": ("舰岛", "ISLAND_OF"),
        "Deck_Feature": ("甲板", "DECK_OF"),
        "Mast_Feature": ("桅杆", "MAST_OF"),
    }

    non_visual_schema = {
        "Length_Overall": ("舰总长", "LENGTH_OVERALL_OF"),
        "Beam": ("舷宽", "BEAM_OF"),
        "Flight_Deck_Width": ("飞行甲板宽", "FLIGHT_DECK_WIDTH_OF"),
        "Draft": ("吃水深度", "DRAFT_OF"),
        "Standard_Displacement": ("标准排水量", "STANDARD_DISPLACEMENT_OF"),
        "Full_Load_Displacement": ("满载排水量", "FULL_LOAD_DISPLACEMENT_OF"),
        "Speed": ("航速", "SPEED_OF"),
        "Range": ("续航力", "RANGE_OF"),
        "Crew": ("舰员编制", "CREW_OF"),
        "Aircraft_Capacity": ("舰载机数量", "AIRCRAFT_CAPACITY_OF"),
        "Power_Output": ("推进功率", "POWER_OUTPUT_OF"),
        "Propulsion": ("推进装置", "PROPULSION_OF"),
        "Flight_Deck_Area": ("飞行甲板面积", "FLIGHT_DECK_AREA_OF"),
        "Island_Position": ("舰岛位置", "ISLAND_POSITION_OF"),
    }

    ship_scores = defaultdict(float)
    ship_evidence = defaultdict(list)
    unmatched_conditions = []

    total_conditions = 0

    # ==================== 1. 装备精确匹配 ====================
    for input_type, items in equipment.items():
        if not items:
            continue

        if input_type not in equipment_schema:
            continue

        entity_type, rel_type = equipment_schema[input_type]

        for raw_item in items:
            total_conditions += 1

            item = normalize_observed_equipment_name(raw_item, entity_type)

            if not item:
                unmatched_conditions.append({
                    "type": input_type,
                    "raw_value": raw_item,
                    "reason": "无法映射到图谱标准实体"
                })
                continue

            if item not in G.nodes:
                unmatched_conditions.append({
                    "type": input_type,
                    "raw_value": raw_item,
                    "normalized": item,
                    "reason": "图谱中不存在该实体"
                })
                continue

            matched_any = False

            for _, config_name, edge_data in G.out_edges(item, data=True):
                edge_rel = get_edge_relation_type(edge_data)

                if edge_rel != rel_type:
                    continue

                ship = get_ship_from_configuration(G, config_name)

                if not ship:
                    continue

                matched_any = True
                ship_scores[ship] += 1.0
                ship_evidence[ship].append({
                    "condition_type": input_type,
                    "input_value": raw_item,
                    "matched_entity": item,
                    "matched_path": f"{item} → {config_name} ← {ship}",
                    "relation_type": rel_type
                })

            if not matched_any:
                unmatched_conditions.append({
                    "type": input_type,
                    "raw_value": raw_item,
                    "normalized": item,
                    "reason": "图谱中没有找到该装备连接到任何舰船配置"
                })

    # ==================== 2. 视觉属性匹配 ====================
    for key, value in visual.items():
        if not value or value in {"未知", "不确定"}:
            continue

        if key not in visual_schema:
            continue

        total_conditions += 1

        fixed_node, rel_type = visual_schema[key]

        if fixed_node not in G.nodes:
            continue

        matched_any = False

        for _, ship, edge_data in G.out_edges(fixed_node, data=True):
            edge_rel = get_edge_relation_type(edge_data)
            desc = str(edge_data.get("description", ""))

            if edge_rel == rel_type and value in desc:
                matched_any = True
                ship_scores[ship] += 0.5
                ship_evidence[ship].append({
                    "condition_type": key,
                    "input_value": value,
                    "matched_entity": fixed_node,
                    "matched_path": f"{fixed_node} → {ship}",
                    "relation_type": rel_type
                })

        if not matched_any:
            unmatched_conditions.append({
                "type": key,
                "raw_value": value,
                "reason": "未在边 description 中匹配到该视觉属性值"
            })

    # ==================== 3. 非视觉属性匹配 ====================
    for key, value in non_visual.items():
        if not value or value == "未知":
            continue

        if key not in non_visual_schema:
            continue

        total_conditions += 1

        fixed_node, rel_type = non_visual_schema[key]

        if fixed_node not in G.nodes:
            continue

        matched_any = False

        for _, ship, edge_data in G.out_edges(fixed_node, data=True):
            edge_rel = get_edge_relation_type(edge_data)
            desc = str(edge_data.get("description", ""))

            if edge_rel == rel_type and value in desc:
                matched_any = True
                ship_scores[ship] += 0.5
                ship_evidence[ship].append({
                    "condition_type": key,
                    "input_value": value,
                    "matched_entity": fixed_node,
                    "matched_path": f"{fixed_node} → {ship}",
                    "relation_type": rel_type
                })

        if not matched_any:
            unmatched_conditions.append({
                "type": key,
                "raw_value": value,
                "reason": "未在边 description 中匹配到该属性值"
            })

    # ==================== 4. 输出结果 ====================
    if not ship_scores:
        return json.dumps({
            "matched_candidates": [],
            "match_level": "no_match",
            "unmatched_conditions": unmatched_conditions,
            "suggestion": "图谱中未找到满足输入条件的舰船。建议检查用户输入是否包含词典外实体，或转入图像识别流程。"
        }, ensure_ascii=False, indent=2)

    max_score = max(ship_scores.values())

    candidates = []

    for ship, score in sorted(ship_scores.items(), key=lambda x: x[1], reverse=True):
        confidence = score / max(total_conditions, 1)

        candidates.append({
            "hull_number": ship,
            "confidence": round(confidence, 4),
            "score": score,
            "matched_evidence": ship_evidence[ship]
        })

    if max_score >= 1.0:
        match_level = "high_confidence"
    elif max_score >= 0.5:
        match_level = "low_confidence"
    else:
        match_level = "very_low_confidence"

    result = {
        "matched_candidates": candidates,
        "match_level": match_level,
        "unmatched_conditions": unmatched_conditions,
        "suggestion": "该结果由 GraphML 精确路径匹配生成，未依赖 LLM 推断。"
    }

    return json.dumps(result, ensure_ascii=False, indent=2)























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

    0. **【最高优先级】严格基于图谱数据——违反即为严重错误**
       这是所有规则中最重要的规则，必须无条件遵守。

       - 你只能使用知识图谱中**实际存在**的数据。
       - **绝对禁止**根据舰船级别、同型舰、或任何其他理由来**推断**某艘舰的属性。
       - **绝对禁止**假设"同级别的舰船属性相同"。
       - 在填写每艘候选舰的 key_attributes 时，你必须能够在知识图谱中找到**直接关联该舰**的证据。

       **违规示例（这是错误的，绝对不要这样做）**：
       - 知识图谱中只有 CVN-68 有 GUN_OF → 勃朗宁 M2，但你给 CVN-69 也列出了"weapon: 勃朗宁 M2"。
         → 这是严重错误！CVN-69 在知识图谱中没有这个关联。

       **正确示例（这才是对的）**：
       - 只给 CVN-68 列出 key_attributes 包含 "weapon: 勃朗宁 M2"。
       - 其他舰要么不列入候选，要么在 differences 中明确标注"知识图谱中未装备该武器"。

       **自查问题**：在输出每艘候选舰之前，问自己："知识图谱中，这艘舰**真的有**这个属性吗？"
       如果答案是"不确定"或"没有"，就不要列出该属性。

    1. **唯一性特征优先**：如果已知条件中包含具体的装备型号（如"勃朗宁 M2 重机枪"），这属于高区分度特征。只有真正装备了该型号的舰船才能作为候选，未装备该型号的舰船**直接排除**，不能因为"同级别"而列入。

    2. **通用特征综合评分**：对于舰型、排水量范围等通用特征，允许模糊匹配。

    3. **逐一比对**：必须对每一个已知条件在知识图谱中进行验证。

    4. **排除规则**：如果候选舰船的某个属性与已知条件明确冲突，必须在 differences 中说明。

    ## 知识图谱数据
    {{context_data}}

    请输出符合要求的匹配结果。
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
    WORKING_DIR = "./ship_index"
    NAVAL_DATA_PATH = "./naval_data.txt"

    # True  = naval_data 用 Python 规则构图，不调用 LLM，不走 ainsert
    # False = 旧模式：仍然走 nano-graphrag 的 ainsert
    USE_RULE_BASE_GRAPH = True

    # naval_data 改了，就设 True 重新构图
    # 图谱稳定后，只测试解析/匹配时改成 False
    REBUILD_INDEX = True

    graphml_file = os.path.join(WORKING_DIR, "graph_chunk_entity_relation.graphml")

    if USE_RULE_BASE_GRAPH:
        # ==================== 规则构图模式 ====================
        # 这个模式用于你的半结构化 naval_data
        # 不调用 LLM，不走 graph_func.ainsert()

        if REBUILD_INDEX and os.path.exists(WORKING_DIR):
            import shutil
            print(f"[重建规则图谱] 删除旧索引目录: {WORKING_DIR}")
            shutil.rmtree(WORKING_DIR)

        if REBUILD_INDEX or not os.path.exists(graphml_file):
            print("[规则构图] 使用 naval_data 直接构建基础知识图谱")

            # 1. 从 naval_data 直接生成 GraphML
            repair_graph_from_naval_data(WORKING_DIR, NAVAL_DATA_PATH)

            # 2. 对规则构图结果做一次 schema 校验和方向规范化
            normalize_graph_directions(WORKING_DIR, NAVAL_DATA_PATH)

        else:
            print(f"[跳过规则构图] 已存在 GraphML，直接使用现有图谱: {graphml_file}")

    else:
        # ==================== nano-graphrag 抽图模式 ====================
        # 这个模式保留给以后：
        # 1. 你想重新测试 nano-graphrag 对 naval_data 的自动抽图能力
        # 2. 或者未来处理非结构化文本资料时，另行改造成 patch_graph 流程

        graph_func = build_graph_rag(WORKING_DIR)

        if REBUILD_INDEX or not os.path.exists(graphml_file):
            try:
                with open(NAVAL_DATA_PATH, "r", encoding="utf-8") as f:
                    await graph_func.ainsert(f.read())

            except Exception as e:
                print(f"[建图失败] ainsert 阶段失败：{type(e).__name__}: {e}")

                if os.path.exists(WORKING_DIR):
                    import shutil
                    print(f"[建图失败] 删除不完整索引目录: {WORKING_DIR}")
                    shutil.rmtree(WORKING_DIR)

                raise e
        else:
            print(f"[跳过建图] 已存在 GraphML，直接使用现有图谱: {graphml_file}")

        # nano-graphrag 抽图后，才需要走这一套：
        # 第一次修方向 → 根据 naval_data 补全 → 第二次修方向
        normalize_graph_directions(WORKING_DIR, NAVAL_DATA_PATH)
        repair_graph_from_naval_data(WORKING_DIR, NAVAL_DATA_PATH)
        normalize_graph_directions(WORKING_DIR, NAVAL_DATA_PATH)

    # ==================== 重新加载 GraphRAG ====================
    # 无论规则构图还是 nano-graphrag 抽图，最后都重新加载当前 GraphML
    graph_func = build_graph_rag(WORKING_DIR)

    # ==================== 检查和打印 ====================
    sanity_check_graph(graph_func.working_dir)
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
            "visual": {
                k: v.get("normalized", "未知")
                for k, v in parsed["observed_attributes"]["visual"].items()
            },
            "non_visual": parsed["observed_attributes"]["non_visual"],
            "equipment_mentioned": parsed["observed_attributes"]["equipment_mentioned"]
        }
        observed_json = json.dumps(observed_for_match, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"解析JSON失败: {e}")
        return

    print("\n" + "=" * 60)
    print("【步骤2】图谱匹配结果：")
    match_result = match_candidates_precise(graph_func.working_dir, observed_json)
    print(match_result)



if __name__ == "__main__":
    asyncio.run(main())
