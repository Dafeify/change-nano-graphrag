# 1. 彻底修复 Windows 控制台编码问题
import sys
import os
import json
import io
import asyncio
import httpx._models
import numpy as np

# 强制 Python 进入 UTF-8 模式
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LANG"] = "en_US.UTF-8"
os.environ["LC_ALL"] = "en_US.UTF-8"

# 2. 修补 httpx 的 header 编码函数，使其支持 UTF-8
_original_normalize = httpx._models._normalize_header_value
def _utf8_normalize(value, encoding=None):
    if isinstance(value, bytes):
        return value
    # 直接使用 UTF-8 编码，而不是 ASCII
    return value.encode("utf-8")
httpx._models._normalize_header_value = _utf8_normalize

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# Windows 下使用更稳定的 Selector 事件循环
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# ==================== 本地 Embedding 模型 ====================
from sentence_transformers import SentenceTransformer

# 加载中文优化的 Embedding 模型（首次运行会自动下载约 1.3GB）
EMBED_MODEL = SentenceTransformer("BAAI/bge-large-zh-v1.5")

# 获取模型的向量维度和最大 token 长度
_embedding_dim = EMBED_MODEL.get_sentence_embedding_dimension()   # 1024
_max_token_size = EMBED_MODEL.max_seq_length                     # 512

def local_embedding(texts):
    """
    本地 Embedding 函数
    """

    embeddings = EMBED_MODEL.encode(
        texts,
        normalize_embeddings=True,
        batch_size=4,
        show_progress_bar=False,
        convert_to_numpy=True
    )

    return embeddings
# 手动挂载 Nano-GraphRAG 必需属性
local_embedding.embedding_dim = _embedding_dim
local_embedding.max_token_size = _max_token_size

# ==================== 统一的 Embedding 接口（可切换） ====================
async def get_embeddings(texts: list[str]) -> np.ndarray:
    """
    统一 Embedding 接口。
    当前使用本地 BGE 模型，未来切换远程 API 只需修改此函数。
    """
    # --- 当前：本地模型 ---
    return await local_embedding(texts)
    # --- 未来：远程硅基流动 API（替换上面一行） ---
    # client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    # resp = await client.embeddings.create(model="BAAI/bge-large-zh-v1.5", input=texts)
    # sorted_data = sorted(resp.data, key=lambda x: x.index)
    # return np.array([d.embedding for d in sorted_data])


# ==================== LLM 配置与调用函数 ====================
from openai import AsyncOpenAI
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash

# --- 模型配置 ---
MODEL = "deepseek-ai/DeepSeek-V3"
BASE_URL = "https://api.siliconflow.cn/v1"
API_KEY = ""  # 临时硬编码，测试完记得改回环境变量

# --- 自定义大模型调用函数 ---
async def siliconflow_llm_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    **kwargs
) -> str:

    history_messages = history_messages or []

    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=180.0)
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
        temperature=kwargs.get("temperature", 0.0),
        **kwargs
    )
    result = response.choices[0].message.content

    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": MODEL}})

    return result


# ==================== 纯文本解析模块（修正版：精确区分确定、不确定、未知） ====================
async def direct_text_parse(user_text: str) -> str:
    """只根据用户输入文本提取属性，将口语化描述映射为标准术语，同时区分确定、不确定、未知"""

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



# ==================== 读取 GraphML 图文件，提取 entity_type 属性（或者 label 等其他字段），更新到 vdb_entities.json 中的每个实体上。 ====================
def fix_entity_types_from_graphml(working_dir):
    """
    从 graph_chunk_entity_relation.graphml 中提取每个节点的 entity_type，
    并回写到 vdb_entities.json 中。
    GraphML 节点 ID 就是实体名，需要通过名称匹配。
    """
    import networkx as nx

    entities_file = os.path.join(working_dir, "vdb_entities.json")
    graphml_file = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")

    # 1. 读取实体文件
    if not os.path.exists(entities_file):
        print("实体文件不存在，无法修复实体类型。")
        return False

    with open(entities_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    entities_list = raw.get("data", [])
    if not entities_list:
        print("实体列表为空。")
        return False

    # 2. 读取 GraphML 图文件，建立 实体名 → entity_type 映射
    if not os.path.exists(graphml_file):
        print("GraphML 文件不存在，无法修复实体类型。")
        return False

    G = nx.read_graphml(graphml_file)
    name_to_type = {}

    for node_id, node_data in G.nodes(data=True):
        # node_id 就是实体名（如 "尼米兹级航空母舰"）
        raw_type = node_data.get("entity_type", None)
        if raw_type:
            # 清洗：去掉外层引号，转为标准格式（首字母大写，其余小写）
            clean_type = raw_type.strip('"').lower()
            # 将 snake_case 转为 PascalCase 风格的单词首字母大写
            clean_type = clean_type.replace("_", " ").title().replace(" ", "_")
            name_to_type[node_id] = clean_type

    if not name_to_type:
        print("未能从 GraphML 中提取到任何实体类型。")
        return False

    # 3. 更新实体列表（通过 entity_name 匹配）
    updated = 0
    for ent in entities_list:
        ent_name = ent.get("entity_name", "")
        # 清洗实体名（去掉外层引号）
        clean_name = ent_name.strip('"')
        if clean_name in name_to_type:
            ent["entity_type"] = name_to_type[clean_name]
            updated += 1
        elif ent_name in name_to_type:
            ent["entity_type"] = name_to_type[ent_name]
            updated += 1

    # 4. 写回文件
    raw["data"] = entities_list
    with open(entities_file, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    print(f"已从 GraphML 修复 {updated}/{len(entities_list)} 个实体的类型信息。")
    return True



# ==================== 实体增强：生成嵌入 ====================
async def generate_entity_descriptions(graph_func):
    """读取 vdb_entities.json，为每个实体生成嵌入并写回。"""
    entities_file = os.path.join(graph_func.working_dir, "vdb_entities.json")
    try:
        with open(entities_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        print(f"实体文件未找到：{entities_file}，跳过嵌入生成。")
        return

    # 实际结构：{"embedding_dim":..., "data": [{ "__id__":..., "entity_name":..., ... }, ...]}
    entities_list = raw.get("data", [])
    if not entities_list:
        print("实体列表为空，跳过嵌入生成。")
        return

    print(f"正在为 {len(entities_list)} 个实体生成嵌入...")
    texts = []
    for ent in entities_list:
        # 用 entity_name 构造描述文本，后续可加入更多字段
        name = ent.get("entity_name", "未知实体")
        # 如果有 description 字段也可使用，目前 nano-graphrag 可能没有
        desc = ent.get("description", f"{name}，类型为{ent.get('entity_type', '未知')}")
        texts.append(desc)

    if texts:
        print("开始生成 embedding...")

        embeddings = await get_embeddings(texts)

        for i, ent in enumerate(entities_list):
            ent["embedding"] = embeddings[i].tolist()
        # 写回原文件
        with open(entities_file, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
        print("实体嵌入生成并保存完毕。")


# ==================== 保留轻量级别名替换（仅 Ship_Instance 和 Ship_Class） ====================
def normalize_entity_names(working_dir):
    """
    轻量级实体名称规范化，处理：
    1. Ship_Instance 别名统一（纯舷号、纯舰名、人名 → 舷号+舰名）
    2. Ship_Class 别名统一（后缀裁剪）
    3. 装备实体去重（功能描述后缀 → 纯型号名）
    4. 移除误分类的无效 Ship_Instance（地名、日期等）
    5. 清理冗余 Propulsion（如仅有"推进"时保留，有"4轴推进"时移除"推进"）
    """
    import re

    entities_file = os.path.join(working_dir, "vdb_entities.json")
    with open(entities_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # ========== Ship_Instance 别名映射表 ==========
    SHIP_ALIAS = {
        # 纯舷号 → 舷号+舰名
        "CVN-68": "CVN-68 尼米兹号",
        "CVN-69": "CVN-69 艾森豪威尔号",
        "CVN-70": "CVN-70 卡尔文森号",
        "CVN-71": "CVN-71 罗斯福号",
        "CVN-72": "CVN-72 林肯号",
        "CVN-73": "CVN-73 华盛顿号",
        "CVN-74": "CVN-74 斯坦尼斯号",
        "CVN-75": "CVN-75 杜鲁门号",
        "CVN-76": "CVN-76 里根号",
        "CVN-77": "CVN-77 布什号",
        # 纯舰名 → 舷号+舰名
        "尼米兹号": "CVN-68 尼米兹号",
        "尼米兹号航空母舰": "CVN-68 尼米兹号",
        "USS NIMITZ CVN-68": "CVN-68 尼米兹号",
        "艾森豪威尔号": "CVN-69 艾森豪威尔号",
        "USS DWIGHT D. EISENHOWER": "CVN-69 艾森豪威尔号",
        "德怀特·艾森豪威尔号": "CVN-69 艾森豪威尔号",
        "卡尔文森号": "CVN-70 卡尔文森号",
        "卡尔·文森号": "CVN-70 卡尔文森号",
        "CVN-70 卡尔·文森号": "CVN-70 卡尔文森号",
        "罗斯福号": "CVN-71 罗斯福号",
        "西奥多·罗斯福号": "CVN-71 罗斯福号",
        "罗斯福号航空母舰": "CVN-71 罗斯福号",
        "林肯号": "CVN-72 林肯号",
        "亚伯拉罕·林肯号": "CVN-72 林肯号",
        "林肯号航空母舰": "CVN-72 林肯号",
        "华盛顿号": "CVN-73 华盛顿号",
        "乔治·华盛顿号": "CVN-73 华盛顿号",
        "华盛顿号航空母舰": "CVN-73 华盛顿号",
        "斯坦尼斯号": "CVN-74 斯坦尼斯号",
        "约翰·斯坦尼斯号": "CVN-74 斯坦尼斯号",
        "杜鲁门号": "CVN-75 杜鲁门号",
        "哈里·杜鲁门号": "CVN-75 杜鲁门号",
        "哈里·S·杜鲁门": "CVN-75 杜鲁门号",
        "里根号": "CVN-76 里根号",
        "罗纳德·里根": "CVN-76 里根号",
        "罗纳德·里根号": "CVN-76 里根号",
        "布什号": "CVN-77 布什号",
        "乔治·布什号": "CVN-77 布什号",
        "乔治·H·W·布什": "CVN-77 布什号",
        "布什号航空母舰": "CVN-77 布什号",
    }

    # ========== Ship_Class 别名映射表 ==========
    CLASS_ALIAS = {
        "尼米兹级航空母舰": "尼米兹级",
        "尼米兹级核动力航母": "尼米兹级",
        "尼米兹级航母": "尼米兹级",
        "尼米兹级核动力航空母舰": "尼米兹级",
        "USS NIMITZ": "尼米兹级",
        "NIMITZ CLASS": "尼米兹级",
        "尼米兹": "尼米兹级",
    }

    # ========== 装备实体规范化映射表 ==========
    EQUIPMENT_ALIAS = {
        # 雷达系统
        "AN/SPS-48E 3D空中搜索雷达": "AN/SPS-48E",
        "AN/SPS-49(V)5 2D空中搜索雷达": "AN/SPS-49(V)5",
        "AN/SPQ-9B目标截获雷达": "AN/SPQ-9B",
        "AN/SPN-46空中管制雷达": "AN/SPN-46",
        "AN/SPN-43C空中管制雷达": "AN/SPN-43C",
        "AN/SPN-41着陆辅助雷达": "AN/SPN-41",
        "MK 91 NSSM引导系统": "MK-91 NSSM",
        "MK 95雷达": "MK-95",
        "MK 91 NSSM": "MK-91 NSSM",
        "MK 95": "MK-95",
        # 电子战系统
        "AN/SLQ-32A(V)4电战反制系统": "AN/SLQ-32A(V)4",
        "AN/SLQ-25A NIXIE鱼雷反制系统": "AN/SLQ-25A NIXIE",
        "AN/SLQ-25A Nixie鱼雷反制系统": "AN/SLQ-25A NIXIE",
        "AN/SLQ-25A NIXIE": "AN/SLQ-25A NIXIE",
        # 武器系统
        "RIM-7海麻雀短程防空导弹发射器": "RIM-7海麻雀导弹",
        "RIM-7海麻雀导弹发射器": "RIM-7海麻雀导弹",
        "RIM-7海麻雀导弹": "RIM-7海麻雀导弹",
        "RIM-116拉姆短程防空导弹系统": "RIM-116拉姆导弹系统",
        "RIM-116拉姆导弹系统": "RIM-116拉姆导弹系统",
        "密集阵近程防御武器系统": "密集阵MK-15",

        "可搭载":"舰载机容量",
        "球鼻艏":"球鼻首",
        "西屋A4W":"西屋A4W压水核反应堆",
        "推进系统":"4轴推进",
        "RIM-116拉姆":"RIM-116拉姆导弹系统",
        "RIM-116拉姆导弹": "RIM-116拉姆导弹系统",
        "A-6": "RIM-7海麻雀导弹",
        "RIM-7海麻雀": "A-6 攻击机",
        "A-6E": "A-6E攻击机",
        "A-6E入侵者": "A-6E攻击机",
        "A-6E“入侵者”攻击机": "A-6E攻击机",
        "C-2“灰狗”运输机": "C-2运输机",
        "C-2快轮运输机": "C-2运输机",
        "E-2": "E-2 预警机",
        "E - 2空中预警机": "E-2 预警机",
        "E-2 鹰眼": "E-2 预警机",
        "E-2C": "E-2C 空中预警机",
        "E-2C “鹰眼” 预警机": "E-2C 空中预警机",
        "E-2C“鹰眼”空中早期预警机": "E-2C 空中预警机",
        "E-2C“鹰眼”预警机": "E-2C 空中预警机",
        "E-2C空中预警机": "E-2C 空中预警机",
        "E-2C预警机系统": "E-2C 空中预警机",
        "E-2C鹰眼": "E-2C 空中预警机",
        "E-2C鹰眼式预警机": "E-2C 空中预警机",




    }

    # ========== 需要移除的无效 Ship_Instance ==========
    INVALID_SHIP_INSTANCE_NAMES = {
        "圣迭戈北岛海军基地",
        "诺福克海军基地",
        "华盛顿州基察普海军基地",
        "纽波特纽斯造船厂",
        "乔治·华盛顿",
        "约翰·C·斯坦尼斯",
    }

    DATE_PATTERN = re.compile(r'^\d{4}年\d{1,2}月\d{1,2}日$|^\d{4}年$')

    # ========== 开始处理 ==========
    entities_to_remove = []

    for ent in raw["data"]:
        name = ent.get("entity_name", "").strip('"')
        etype = ent.get("entity_type", "unknown").lower()

        # --- Ship_Instance ---
        if etype == "ship_instance":
            # 日期格式的 Ship_Instance 直接移除
            if DATE_PATTERN.match(name):
                ent["_remove"] = True
                entities_to_remove.append(f"{name}(日期)")
                continue
            # 地名等无效名称移除
            if name in INVALID_SHIP_INSTANCE_NAMES:
                ent["_remove"] = True
                entities_to_remove.append(f"{name}(无效)")
                continue
            # 别名映射
            if name in SHIP_ALIAS:
                ent["entity_name"] = SHIP_ALIAS[name]

        # --- Ship_Class ---
        elif etype == "ship_class":
            if name in CLASS_ALIAS:
                ent["entity_name"] = CLASS_ALIAS[name]

        # --- 装备实体 ---
        elif etype in ("radar_system", "countermeasure_system", "combat_system",
                       "weapon_system", "shipboard_gun"):
            if name in EQUIPMENT_ALIAS:
                ent["entity_name"] = EQUIPMENT_ALIAS[name]

    # ========== 清理冗余 Propulsion ==========
    has_4shaft = any(
        e.get("entity_name", "").strip('"') == "4轴推进"
        for e in raw["data"]
        if e.get("entity_type", "unknown").lower() == "propulsion"
    )
    for ent in raw["data"]:
        if ent.get("entity_type", "unknown").lower() == "propulsion":
            name = ent.get("entity_name", "").strip('"')
            if name in ("推进", "推进装置") and has_4shaft:
                ent["_remove"] = True

    # ========== 执行删除 ==========
    raw["data"] = [e for e in raw["data"] if not e.get("_remove", False)]

    with open(entities_file, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    print(f"实体规范化完成，移除了 {len(entities_to_remove)} 个无效实体。")


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

# ==================== 打印实体检查重复 ====================
def inspect_entities(working_dir):
    entities_file = os.path.join(working_dir, "vdb_entities.json")
    with open(entities_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    entities_list = raw.get("data", [])

    from collections import defaultdict
    type_groups = defaultdict(list)
    for ent in entities_list:
        etype = ent.get("entity_type", "unknown")
        name = ent.get("entity_name", "").strip('"')
        type_groups[etype].append(name)

    print("\n========== 知识图谱实体清单 ==========")
    for etype, names in sorted(type_groups.items()):
        unique_names = sorted(set(names))
        print(f"\n【{etype}】({len(names)} 个, 去重后 {len(unique_names)} 个)")
        for n in unique_names:
            print(f"  - {n}")
    print("========================================\n")



# --- 主程序 ---
async def main():
    graph_func = GraphRAG(
        working_dir="./ship_index",
        best_model_func=siliconflow_llm_complete,
        cheap_model_func=siliconflow_llm_complete,
        best_model_id=MODEL,
        cheap_model_id=MODEL,
        embedding_func=local_embedding,
        best_model_max_async=2,
        cheap_model_max_async=2,

        chunk_token_size=600,
        chunk_overlap_token_size=80,

        entity_extract_max_gleaning=1,
    )

    # ========== 首次运行时构建知识图谱 ==========
    if not os.path.exists("./ship_index/vdb_entities.json"):
        print("开始构建知识图谱...")

        with open("./naval_data.txt", "r", encoding='utf-8') as f:
            content = f.read()

        # 先只插入前 3000 字符测试
        await graph_func.ainsert(content[:3000])

        print("知识图谱构建完成")

    # ========== 实体类型修复与增强 ==========
    if fix_entity_types_from_graphml(graph_func.working_dir):
        #await generate_entity_descriptions(graph_func)
        # dedicated_entity_linker(graph_func.working_dir)  # 已弃用
        normalize_entity_names(graph_func.working_dir)
        inspect_entities(graph_func.working_dir)
    else:
        print("实体类型修复失败，跳过实体链接。")

    # ---------- 测试 ----------
    user_text = (
        "一艘很大的航母，装备勃朗宁 M2 重机枪"
    )
    print("=" * 60)
    print("【步骤1】纯文本解析结果：")
    parse_result = await direct_text_parse(user_text)
    print(parse_result)

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