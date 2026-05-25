"""
Reference:
 - Prompts are from [graphrag](https://github.com/microsoft/graphrag)
"""

GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}

PROMPTS[
    "claim_extraction"
] = """-Target activity-
You are an intelligent assistant that helps a human analyst to analyze claims against certain entities presented in a text document.

-Goal-
Given a text document that is potentially relevant to this activity, an entity specification, and a claim description, extract all entities that match the entity specification and all claims against those entities.

-Steps-
1. Extract all named entities that match the predefined entity specification. Entity specification can either be a list of entity names or a list of entity types.
2. For each entity identified in step 1, extract all claims associated with the entity. Claims need to match the specified claim description, and the entity should be the subject of the claim.
For each claim, extract the following information:
- Subject: name of the entity that is subject of the claim, capitalized. The subject entity is one that committed the action described in the claim. Subject needs to be one of the named entities identified in step 1.
- Object: name of the entity that is object of the claim, capitalized. The object entity is one that either reports/handles or is affected by the action described in the claim. If object entity is unknown, use **NONE**.
- Claim Type: overall category of the claim, capitalized. Name it in a way that can be repeated across multiple text inputs, so that similar claims share the same claim type
- Claim Status: **TRUE**, **FALSE**, or **SUSPECTED**. TRUE means the claim is confirmed, FALSE means the claim is found to be False, SUSPECTED means the claim is not verified.
- Claim Description: Detailed description explaining the reasoning behind the claim, together with all the related evidence and references.
- Claim Date: Period (start_date, end_date) when the claim was made. Both start_date and end_date should be in ISO-8601 format. If the claim was made on a single date rather than a date range, set the same date for both start_date and end_date. If date is unknown, return **NONE**.
- Claim Source Text: List of **all** quotes from the original text that are relevant to the claim.

Format each claim as (<subject_entity>{tuple_delimiter}<object_entity>{tuple_delimiter}<claim_type>{tuple_delimiter}<claim_status>{tuple_delimiter}<claim_start_date>{tuple_delimiter}<claim_end_date>{tuple_delimiter}<claim_description>{tuple_delimiter}<claim_source>)

3. Return output in English as a single list of all the claims identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

-Examples-
Example 1:
Entity specification: organization
Claim description: red flags associated with an entity
Text: According to an article on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B. The company is owned by Person C who was suspected of engaging in corruption activities in 2015.
Output:

(COMPANY A{tuple_delimiter}GOVERNMENT AGENCY B{tuple_delimiter}ANTI-COMPETITIVE PRACTICES{tuple_delimiter}TRUE{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}Company A was found to engage in anti-competitive practices because it was fined for bid rigging in multiple public tenders published by Government Agency B according to an article published on 2022/01/10{tuple_delimiter}According to an article published on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B.)
{completion_delimiter}

Example 2:
Entity specification: Company A, Person C
Claim description: red flags associated with an entity
Text: According to an article on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B. The company is owned by Person C who was suspected of engaging in corruption activities in 2015.
Output:

(COMPANY A{tuple_delimiter}GOVERNMENT AGENCY B{tuple_delimiter}ANTI-COMPETITIVE PRACTICES{tuple_delimiter}TRUE{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}Company A was found to engage in anti-competitive practices because it was fined for bid rigging in multiple public tenders published by Government Agency B according to an article published on 2022/01/10{tuple_delimiter}According to an article published on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B.)
{record_delimiter}
(PERSON C{tuple_delimiter}NONE{tuple_delimiter}CORRUPTION{tuple_delimiter}SUSPECTED{tuple_delimiter}2015-01-01T00:00:00{tuple_delimiter}2015-12-30T00:00:00{tuple_delimiter}Person C was suspected of engaging in corruption activities in 2015{tuple_delimiter}The company is owned by Person C who was suspected of engaging in corruption activities in 2015)
{completion_delimiter}

-Real Data-
Use the following input for your answer.
Entity specification: {entity_specs}
Claim description: {claim_description}
Text: {input_text}
Output: """

PROMPTS[
    "community_report"
] = """你是一个军舰知识分析专家。请根据提供的实体和关系，生成一个型号属性映射表。

映射表应以**特征组合**为索引，每个组合下列出满足该组合的候选舷号（Ship_Instance）。

示例格式：
{{
  "特征组合": [
    {{
      "条件": {{"雷达": "AN/SPY-6", "近防系统": "Phalanx Block 1B", "舰岛层数": "3"}},
      "候选舷号": ["CVN-72", "CVN-73"]
    }},
    {{
      "条件": {{"雷达": "AN/SPY-6", "近防系统": "SeaRAM", "舰岛层数": "4"}},
      "候选舷号": ["CVN-76", "CVN-77"]
    }}
  ]
}}

关键要求：
1. 特征组合中的条件应使用实体类型的中文简称：雷达(Radar_System)、对抗系统(Countermeasure_System)、指挥作战(Combat_System)、武器装备(Weapon_System)、舰载火炮(Shipboard_Gun)、动力装置(Powerplant)、舰载飞机(Aircraft)、舰首(Bow)、舰尾(Stern)、舰岛(Island)、甲板(Deck)、桅杆(Mast)。
2. 每个特征组合的候选舷号必须是从社区实体中真实存在的 Ship_Instance。
3. 只输出 JSON，不要包含其他解释文字。

使用以下文本生成映射表：
Text:
```
{input_text}
```

Output:
"""


PROMPTS["entity_extraction"] = """-Goal-
Given a text document, identify all entities of the specified types and all relationships among them.

-Steps-
1. 识别所有实体。对于每个识别的实体，提取以下信息：
- entity_name: 必须严格遵循已知实体词典中的规定
- entity_type: 必须是以下类型之一：[{entity_types}]
- entity_description: 根据填写规则写入描述信息

2. 装备聚合规则（Configuration 的使用）：
   - 每艘舰的每一类装备都必须创建一个 Configuration 套件节点
   - 命名格式为"舰名+装备类型+套件"，例如"CVN-73 雷达套件"、"CVN-68 武器套件"
   - 用 EQUIPPED_WITH 关系将舰连接到 Configuration，再用专用关系将 Configuration 连接到每个具体装备实体
   - 专用关系包括：HAS_RADAR, HAS_COUNTERMEASURE, HAS_COMBAT, HAS_COMMUNICATION,
     HAS_DATA_LINK, HAS_WEAPON, HAS_GUN, HAS_AIRCRAFT, HAS_POWERPLANT,
     HAS_CATAPULT_EQUIP, HAS_ARRESTING_EQUIP, HAS_ARMOR

## 已知实体词典（所有实体名必须来自以下列表，不得自创）

### Ship_Class
尼米兹级

### Ship_Instance
CVN-68 尼米兹号, CVN-69 艾森豪威尔号, CVN-70 卡尔文森号, CVN-71 西奥多·罗斯福号,
CVN-72 亚伯拉罕·林肯号, CVN-73 乔治·华盛顿号, CVN-74 约翰·C·斯坦尼斯号,
CVN-75 哈里·S·杜鲁门号, CVN-76 罗纳德·里根号, CVN-77 乔治·H·W·布什号

### Bow
球鼻艏

### Stern
（暂无已知实体名）

### Deck
斜角甲板, 直角甲板

### Island（通用实体名固定为"舰岛位于右舷"）
舰岛位于右舷
- 如有位置细节（如"中部"、"靠近舰艉"），写入 entity_description

### Mast（通用实体名从以下选择）
柱状综合桅杆, 塔状桅杆, 复合桅杆
- 如有外形修饰（如"细长高大"、"与舰岛整合"），写入 entity_description

### Powerplant
A4W 压水核反应堆, A4W/A1G 压水核反应堆, 蒸汽涡轮发动机, 四轴双主舵,
四轴四桨, 四轴五桨, 四桨四轴双舵, 汽轮发电机, 应急柴油发电机, 备用柴油机

### Catapult（通用实体名固定为"弹射器"）
弹射器
- 有型号时 entity_description 写型号（如"C-13-1"），型号未知写"型号未知"

### Arresting_Gear（通用实体名固定）
拦阻索, 拦阻网
- 有型号时 entity_description 写型号（如"Mk 7 Mod 3 型"），型号未知写"型号未知"

### Radar_System
AN/SPS-48C/E, AN/SPS-48E, AN/SPS-49(V)1, AN/SPS-49(V)5, AN/SPS-43A,
AN/SPS-67, AN/SPS-67V, AN/SPS-67V-1, AN/SPQ-9A, AN/SPQ-9B,
AN/SPN-46, AN/SPN-43C, AN/SPN-41, AN/SPN-44,
Mk 91 NSSM, Mk 95, MK91-1, MK-73, SPS-64(V)9, LN-66, URN-25, MK23 TAS

### Radar_Function
对空搜索, 对海搜索, 火控, 空中管制, 目标截获, 导航, 测速

### Countermeasure_System
AN/SLQ-32(V)4, SLY-2, AN/WLR-1H, Mk 36 SRBOC,
AN/SLQ-25, SLQ-25A, SLQ-29, SLQ-36

### Countermeasure_Function
电子战, 电子侦察, 诱饵发射, 拖曳鱼雷诱饵, 电子干扰

### Combat_System
ACDS, ACDS Block 0/1, ACDS Block 1, NTDS, SSDS Mk 2, MK-23 TAS

### Combat_Function
战斗指挥, 战术数据, 舰艇自卫, 目标搜获, 信息指挥

### Communication_System
SRR-1, WSC-3, WSC-6, USC-38, SSQ-82, SQQ-1, JOTS, POST, CVIC,
TESS UMM-1(V)1, JMCIS, SSQ-1A, 全光纤数字化通信系统, IT21, IT‑21 非保密型局域网系统

### Communication_Function
卫星通信, 战术环境支援, 航母情报, 指挥信息系统, 联合战术系统

### Data_Link
LINK-4A, LINK-11, LINK-14, LINK-16

### Weapon_System
Mk 25, Mk 29, Mk 31, Mk 49, Mk 57 Mod 3, RIM-7, RIM-7M, RIM-116, Mk 15,
LOCUST, 三联装324毫米鱼雷发射管

### Weapon_Function
短程防空, 近防系统, 导弹发射装置, 激光武器, 鱼雷发射装置

### Shipboard_Gun
Mk 38, 勃朗宁 M2

### Shipboard_Gun_Function
遥控机炮, 重机枪

### Aircraft
F/A-18E/F, F/A-18C/D, F/A-18A/B/C/D, F/A-18A/C/E, F/A-18F, F/A-18,
F-14, F-14D, F-14A/B/D, F-35C, E-2C, E-2D, E-2, EA-6B, EA-18G, A-6E,
S-3A/B, S-3A, S-3B, ES-3A, SH-3G/H, SH-3G, SH-3H, SH-60F, HH-60H,
MH-60R, MH-60R/S, SH-60, UH-60, C-2, C-2A
- 退役机型 entity_description 写"退役"，现役写"无"

### Aircraft_Function
战斗攻击机, 电子战飞机, 预警机, 反潜机, 侦察机, 运输机, 直升机

### Armor_Protection
双层舰壳, X 形吸能支撑结构, HY-80 高强度钢, 水密隔舱壁, 防火隔壁, 水密隔舱,
纵向防雷舱壁, 凯夫拉装甲, 先进灭火系统, 高强度合金钢, 多层隔离防护结构,
隐身吸波材料, 高弹性钢, 泡沫消防装置, 双层船体, X形构件, 多层隔舱防护, 箱型防御结构

### Shipyard
纽波特纽斯造船厂

### Service_Status
现役

### 纯文本属性（实体名固定为属性描述，数值写入 entity_description）
Length_Overall: "舰总长"
Beam: "舷宽"
Flight_Deck_Width: "飞行甲板宽"
Draft: "吃水深度"
Standard_Displacement: "标准排水量"
Full_Load_Displacement: "满载排水量"
Speed: "航速"
Range: "续航力"
Crew: "舰员编制"
Aircraft_Capacity: "舰载机数量"
Power_Output: "推进功率"
Propulsion: "推进装置"
Flight_Deck_Area: "飞行甲板面积"
Island_Position: "舰岛位置"
Homeport: "母港"

### entity_description 填写规则
- 纯文本属性：写入具体数值
- 弹射器/拦阻装置：写入型号，型号未知写"型号未知"
- 舰岛/桅杆：有细节时写入，无细节写"无"
- 舰载机：退役写"退役"，现役写"无"
- 其他装备实体：写"无"

Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description)

3. 从步骤1中识别出的实体中，找出所有明确相关的 (source_entity, target_entity) 对，提取关系信息。
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

4. Return output in English as a single list. Use **{record_delimiter}** as the list delimiter. When finished, output {completion_delimiter}

-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""


PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""


PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    # 舰船身份
    "Ship_Class", "Ship_Instance",
    # 视觉属性
    "Bow", "Stern", "Deck", "Island", "Mast",
    # 装备系统
    "Radar_System", "Countermeasure_System", "Combat_System",
    "Communication_System", "Data_Link", "Weapon_System",
    "Shipboard_Gun", "Aircraft", "Powerplant", "Catapult", "Arresting_Gear",
    # 功能分类
    "Radar_Function", "Countermeasure_Function", "Combat_Function",
    "Communication_Function", "Weapon_Function", "Aircraft_Function",
    "Shipboard_Gun_Function",
    # 辅助功能与结构
    "Armor_Protection", "Shipyard", "Service_Status",
    # 纯文本属性
    "Length_Overall", "Beam", "Flight_Deck_Width", "Draft",
    "Standard_Displacement", "Full_Load_Displacement",
    "Speed", "Range", "Crew", "Aircraft_Capacity",
    "Power_Output", "Propulsion", "Flight_Deck_Area",
    "Island_Position", "Homeport",
    # 结构节点
    "Configuration",
]
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS[
    "local_rag_response"
] = """---Role---

你是一个军舰识别专家，负责根据观察到的属性在知识图谱中匹配最可能的舰船型号。

---Goal---

请结合检索到的知识图谱信息，找出所有符合或部分符合观察属性的候选舰船。

你必须严格按照以下 JSON Schema 输出：

{{
  "matched_candidates": [
    {{
      "hull_number": "CVN-72",
      "confidence": 0.85,
      "match_points": ["满载排水量104200吨", "球鼻艏", "三层舰岛"],
      "differences": ["雷达型号不匹配(观察为未知)"],
      "key_attributes": {{
        "full_load_displacement": "104200吨",
        "draft": "11.9米",
        "radar": ["AN/SPS-48E", "AN/SPS-49(V)5"],
        "weapon": ["密集阵MK-15"]
      }}
    }}
  ],
  "match_summary": "简要说明匹配逻辑和主要区分点"
}}

---关键要求---
1. 候选列表按置信度降序排列
2. match_points 列出该候选与观察属性匹配的关键点
3. differences 列出该候选与观察属性不一致之处（如观察属性为"未知"，也请注明）
4. key_attributes 提供该候选型号在知识图谱中的关键属性，方便对比
5. 如果没有任何候选满足条件，matched_candidates 为空数组，并在 match_summary 中说明
6. 只输出 JSON，不要任何额外文字

---Data tables---

{context_data}

---Goal---

Generate the matched candidates JSON as specified above.
"""

PROMPTS[
    "global_map_rag_points"
] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response should be JSON formatted as follows:
{{
    "points": [
        {{"description": "Description of point 1...", "score": score_value}},
        {{"description": "Description of point 2...", "score": score_value}}
    ]
}}

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".
Do not include information where the supporting evidence for it is not provided.


---Data tables---

{context_data}

---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".
Do not include information where the supporting evidence for it is not provided.

The response should be JSON formatted as follows:
{{
    "points": [
        {{"description": "Description of point 1", "score": score_value}},
        {{"description": "Description of point 2", "score": score_value}}
    ]
}}
"""

PROMPTS[
    "global_reduce_rag_response"
] = """---Role---

You are a helpful assistant responding to questions about a dataset by synthesizing perspectives from multiple analysts.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}


---Analyst Reports---

{report_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS[
    "naive_rag_response"
] = """You're a helpful assistant
Below are the knowledge you know:
{content_data}
---
If you don't know the answer or if the provided knowledge do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.
---Target response length and format---
{response_type}
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["default_text_separator"] = [
    # Paragraph separators
    "\n\n",
    "\r\n\r\n",
    # Line breaks
    "\n",
    "\r\n",
    # Sentence ending punctuation
    "。",  # Chinese period
    "．",  # Full-width dot
    ".",  # English period
    "！",  # Chinese exclamation mark
    "!",  # English exclamation mark
    "？",  # Chinese question mark
    "?",  # English question mark
    # Whitespace characters
    " ",  # Space
    "\t",  # Tab
    "\u3000",  # Full-width space
    # Special characters
    "\u200b",  # Zero-width space (used in some Asian languages)
]
