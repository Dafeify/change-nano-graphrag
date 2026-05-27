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
Given a structured text document about Nimitz-class aircraft carriers, identify all entities and relationships according to the rules below. The entity_name must NOT include any quotation marks or angle brackets.

-Input Format-
The text is organized in sections marked by 【】 and [ ]:
- 【CVN-XX Name】 marks the beginning of one ship's data
- [SHIP] section contains the ship's identity
- [VISUAL_FEATURES], [TEXT_ATTRIBUTES] sections list attributes as "Key: Value"
- Sections like [RADAR_SYSTEM], [WEAPON_SYSTEM] list equipment names
- [CONFIGURATION] section lists equipment suites
- Sections ending with _FUNCTION map functions to equipment
- [/SHIP] marks the end of one ship's data

-Entity and Relationship Rules-
You MUST use the known entity dictionary and relationship types strictly. Do NOT create any entity names or relationship types outside the given lists. All entity_name values must be output without any brackets or quotes.

## 已知实体词典 (必须使用以下实体名，不得修改)
### Ship_Class
尼米兹级
### Ship_Instance
CVN-68 尼米兹号, CVN-69 艾森豪威尔号, CVN-70 卡尔文森号, CVN-71 西奥多·罗斯福号,
CVN-72 亚伯拉罕·林肯号, CVN-73 乔治·华盛顿号, CVN-74 约翰·C·斯坦尼斯号,
CVN-75 哈里·S·杜鲁门号, CVN-76 罗纳德·里根号, CVN-77 乔治·H·W·布什号
### Bow (实体名固定为 "船首")
船首
### Stern (实体名固定为 "船尾")
船尾
### Deck (实体名固定为 "甲板")
甲板
### Island (实体名固定为 "舰岛")
舰岛
### Mast (实体名固定为 "桅杆")
桅杆
### Powerplant
A4W 压水核反应堆, A4W/A1G 压水核反应堆, 蒸汽涡轮发动机, 四轴双主舵,
四轴四桨, 四轴五桨, 四桨四轴双舵, 汽轮发电机, 应急柴油发电机, 备用柴油机
### Catapult (实体名固定为 "弹射器")
弹射器
### Arresting_Gear (实体名固定为 "拦阻索" 或 "拦阻网")
拦阻索, 拦阻网
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
TESS UMM-1(V)1, JMCIS, SSQ-1A, 全光纤数字化通信系统,  IT-21
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
### Aircraft_Function
战斗攻击机, 电子战飞机, 预警机, 反潜机, 侦察机, 运输机, 直升机
### Armor_Protection
双层舰壳, X 形吸能支撑结构, HY-80 高强度钢, 水密隔舱壁, 防火隔壁, 水密隔舱,
纵向防雷舱壁, 凯夫拉装甲, 先进灭火系统, 高强度合金钢, 多层隔离防护结构,
隐身吸波材料, 高弹性钢, 泡沫消防装置, 双层船体, X形构件, 多层隔舱防护, 箱型防御结构
### Shipyard (实体名固定为 "建造船厂")
建造船厂
### Service_Status (实体名固定为 "服役状态")
服役状态


### Length_Overall (实体名固定为 "舰总长")
舰总长
### Beam (实体名固定为 "舷宽")
舷宽
### Flight_Deck_Width (实体名固定为 "飞行甲板宽")
飞行甲板宽
### Draft (实体名固定为 "吃水深度")
吃水深度
### Standard_Displacement (实体名固定为 "标准排水量")
标准排水量
### Full_Load_Displacement (实体名固定为 "满载排水量")
满载排水量
### Speed (实体名固定为 "航速")
航速
### Range (实体名固定为 "续航力")
续航力
### Crew (实体名固定为 "舰员编制")
舰员编制
### Aircraft_Capacity (实体名固定为 "舰载机数量")
舰载机数量
### Power_Output (实体名固定为 "推进功率")
推进功率
### Propulsion (实体名固定为 "推进装置")
推进装置
### Flight_Deck_Area (实体名固定为 "飞行甲板面积")
飞行甲板面积
### Island_Position (实体名固定为 "舰岛位置")
舰岛位置
### Homeport (实体名固定为 "母港")
母港


### Configuration
- 配置实体名必须严格等于 [CONFIGURATION] 部分中每一行冒号前的完整文本，不得修改或自行拼凑。
- 命名规则为 "舰名+装备类型+套件"，如 "CVN-68 雷达套件"。
- 如果 [CONFIGURATION] 中某行为 "CVN-68 雷达套件:"，则实体名就是 "CVN-68 雷达套件"，不要省略或添加内容。
- 关键：必须为 [CONFIGURATION] 中列表内的所有项创建关系，不得遗漏或只取第一条。例如，当套件包含多个实体时，你必须为列表中的每一个实体都创建一条对应的关系。
- 禁止根据其他章节（如 [RADAR_SYSTEM]）自行推断配置名。
- 【最高优先级禁令】绝对禁止编造、自创任何 [CONFIGURATION] 中不存在的配置实体；原文没有的配置名称，一律不生成。
- 【提取规则】仅逐行提取 [CONFIGURATION] 区块内真实存在的行，无中生有的配置名（如战斗套件、对抗套件）严禁输出。
- 【数量约束】每一艘舰船的 [CONFIGURATION] 有多少行，就生成多少个 Configuration 实体，不多生成、不少生成。

## 已知关系类型 (必须严格模仿以下示例的方向，禁止任何反转)
⚠️ 【方向区分铁律】关系方向必须根据具体关系类型判断，不能只根据 `_OF` 后缀判断。
视觉特征、纯文本属性、辅助结构关系的方向是：属性/特征/结构实体 → Ship_Instance。
装备套件关系的方向是：装备/系统实体 → Configuration。
功能分类关系的方向是：装备/系统实体 → Function。

### 身份归属
INSTANCE_OF 示例: 尼米兹级 → CVN-68 尼米兹号
### 视觉特征
BOW_OF 示例: 船首 → CVN-68 尼米兹号
STERN_OF 示例: 船尾 → CVN-68 尼米兹号
DECK_OF 示例: 甲板 → CVN-68 尼米兹号
ISLAND_OF 示例: 舰岛 → CVN-68 尼米兹号
MAST_OF 示例: 桅杆 → CVN-68 尼米兹号
### 装备套件连接
EQUIPPED_WITH 示例: CVN-68 尼米兹号 → CVN-68 雷达套件
RADAR_OF 示例: AN/SPS-48C/E → CVN-68 雷达套件
WEAPON_OF 示例: Mk 29 → CVN-68 武器套件
COUNTERMEASURE_OF 示例: AN/SLQ-32(V)4 → CVN-68 电子战套件
COMBAT_SYSTEM_OF 示例: ACDS → CVN-68 作战系统套件
COMMUNICATION_OF 示例: SRR-1 → CVN-68 通信套件
DATA_LINK_OF 示例: LINK-4A → CVN-68 数据链套件
GUN_OF 示例: Mk 38 → CVN-68 舰载火炮套件
AIRCRAFT_OF 示例: F/A-18E/F → CVN-68 舰载机联队
POWERPLANT_OF 示例: A4W 压水核反应堆 → CVN-68 动力套件
CATAPULT_OF 示例: 弹射器 → CVN-68 弹射器套件
ARRESTING_GEAR_OF 示例: 拦阻索 → CVN-68 拦阻装置套件
ARMOR_OF 示例: 双层舰壳 → CVN-68 装甲防护套件
### 功能分类连接
HAS_RADAR_FUNCTION 示例: AN/SPS-48C/E → 对空搜索
HAS_COUNTERMEASURE_FUNCTION 示例: AN/SLQ-32(V)4 → 电子战
HAS_COMBAT_FUNCTION 示例: ACDS → 战斗指挥
HAS_COMMUNICATION_FUNCTION 示例: SRR-1 → 卫星通信
HAS_WEAPON_FUNCTION 示例: Mk 29 → 导弹发射装置
HAS_AIRCRAFT_FUNCTION 示例: F/A-18E/F → 战斗攻击机
HAS_SHIPBOARD_GUN_FUNCTION 示例: Mk 38 → 遥控机炮
### 辅助功能与结构连接
ARMOR_PROTECTION_OF 示例: 双层舰壳 → CVN-68 尼米兹号
BUILT 示例: 建造船厂 → CVN-68 尼米兹号
SERVICE_STATUS_OF 示例: 服役状态 → CVN-68 尼米兹号
### 纯文本属性连接
LENGTH_OVERALL_OF 示例: 舰总长 → CVN-68 尼米兹号
BEAM_OF 示例: 舷宽 → CVN-68 尼米兹号
FLIGHT_DECK_WIDTH_OF 示例: 飞行甲板宽 → CVN-68 尼米兹号
DRAFT_OF 示例: 吃水深度 → CVN-68 尼米兹号
STANDARD_DISPLACEMENT_OF 示例: 标准排水量 → CVN-68 尼米兹号
FULL_LOAD_DISPLACEMENT_OF 示例: 满载排水量 → CVN-68 尼米兹号
SPEED_OF 示例: 航速 → CVN-68 尼米兹号
RANGE_OF 示例: 续航力 → CVN-68 尼米兹号
CREW_OF 示例: 舰员编制 → CVN-68 尼米兹号
AIRCRAFT_CAPACITY_OF 示例: 舰载机数量 → CVN-68 尼米兹号
POWER_OUTPUT_OF 示例: 推进功率 → CVN-68 尼米兹号
PROPULSION_OF 示例: 推进装置 → CVN-68 尼米兹号
FLIGHT_DECK_AREA_OF 示例: 飞行甲板面积 → CVN-68 尼米兹号
ISLAND_POSITION_OF 示例: 舰岛位置 → CVN-68 尼米兹号
HOMEPORT_OF 示例: 母港 → CVN-68 尼米兹号

-Steps-
-Steps-
0. **CRITICAL: Handling Parentheses in Entity Names**
   **Dictionary-First Rule**: Before applying any parentheses splitting, 
   check if the ENTIRE text (including parentheses) matches an entity 
   in the known entity dictionary above.
   - If the full text (with parentheses) is found in the dictionary → 
     keep it as the complete entity_name, do NOT split.
     * "AN/SPS-49(V)5" → matches dictionary, entity_name: "AN/SPS-49(V)5"
     * "AN/SLQ-32(V)4" → matches dictionary, entity_name: "AN/SLQ-32(V)4"
     * "TESS UMM-1(V)1" → matches dictionary, entity_name: "TESS UMM-1(V)1"
   
   **Splitting Rule**: If the full text does NOT match any dictionary entry:
   - The entity_name is ONLY the part before the parentheses
   - The part inside parentheses is the entity_description
   - Examples:
     * "弹射器(C-13-1)" → not in dictionary as full text, split to 
       entity_name: "弹射器", entity_description: "C-13-1"
     * "A-6E(退役)" → not in dictionary as full text, split to 
       entity_name: "A-6E", entity_description: "退役"
     * "拦阻索(Mk 7 Mod 3 型)" → not in dictionary as full text, split to 
       entity_name: "拦阻索", entity_description: "Mk 7 Mod 3 型"
   
   This rule applies to ALL sections in the input text.



1. For each section, extract entities as follows:

**Identity sections ([SHIP])**:
- Ship_Instance: entity_name is the value after "Ship_Instance:" (e.g., "CVN-68 尼米兹号")
- Ship_Class: entity_name is the value after "Ship_Class:" (e.g., "尼米兹级")

**Visual features ([VISUAL_FEATURES])**:
- entity_name MUST be the fixed Chinese name: "船首", "船尾", "甲板", "舰岛", "桅杆"
- entity_type is the corresponding type (Bow, Stern, Deck, Island, Mast)
- entity_description is the value after the colon (e.g., "斜角甲板，直角甲板", or "未知")

**Text attributes ([TEXT_ATTRIBUTES])**:
- entity_name MUST be the fixed Chinese attribute name as listed in the dictionary
- entity_type is the corresponding type
- entity_description is the value after the colon

**Equipment sections**:
- Each line under these sections is an entity_name, must be from the dictionary
- entity_type is determined by the section name
- entity_description is "无" by default

**Function sections**:
- The function name (before the colon) is the entity_name, must be from the dictionary
- entity_type is determined by the section name
- entity_description is the list of equipment after the colon

**Configuration section**:
- Each line like "CVN-68 雷达套件:" is a Configuration entity
- entity_name is the line before the colon
- The items listed below are NOT entities, they define relationships
- NO additional Configuration entities allowed: only extract lines that exist in the input text.
- You MUST create all relationship connections for EVERY item listed under each Configuration entity. 
  For example, if a Configuration entity lists 3 items, you must create 3 separate relationships.


**Auxiliary sections**:
- Shipyard: entity_name MUST be "建造船厂", entity_description is the value
- Service_Status: entity_name MUST be "服役状态", entity_description is the value

2. For relationships, create connections by strictly following the type-based rules below.
   These rules apply to ALL ships, regardless of their hull number.

   **Rule 1 - INSTANCE_OF**: Ship_Class → Ship_Instance
     Example: 尼米兹级 → CVN-68 尼米兹号
     Example: 尼米兹级 → CVN-69 艾森豪威尔号
     NEVER reverse.

   **Rule 2 - BOW_OF, STERN_OF, DECK_OF, ISLAND_OF, MAST_OF**:
     Feature_Entity → Ship_Instance
     Example: 船首 → CVN-68 尼米兹号
     Example: 舰岛 → CVN-69 艾森豪威尔号

     INCLUDING CVN-68:
     船首 → CVN-68 尼米兹号 is CORRECT.
     舰岛 → CVN-68 尼米兹号 is CORRECT.
     甲板 → CVN-68 尼米兹号 is CORRECT.
     船尾 → CVN-68 尼米兹号 is CORRECT.
     桅杆 → CVN-68 尼米兹号 is CORRECT.
     CVN-68 尼米兹号 → 船首 is WRONG.
     CVN-68 尼米兹号 → 舰岛 is WRONG.
     CVN-68 尼米兹号 → 甲板 is WRONG.
     CVN-68 尼米兹号 → 船尾 is WRONG.
     CVN-68 尼米兹号 → 桅杆 is WRONG.

     NEVER reverse for ANY ship.

   **Rule 3 - Text attribute relationships**:
     LENGTH_OVERALL_OF, BEAM_OF, FLIGHT_DECK_WIDTH_OF, DRAFT_OF,
     STANDARD_DISPLACEMENT_OF, FULL_LOAD_DISPLACEMENT_OF, SPEED_OF,
     RANGE_OF, CREW_OF, AIRCRAFT_CAPACITY_OF, POWER_OUTPUT_OF,
     PROPULSION_OF, FLIGHT_DECK_AREA_OF, ISLAND_POSITION_OF, HOMEPORT_OF
     must follow:
     Attribute_Entity → Ship_Instance

     Example: 舰总长 → CVN-68 尼米兹号
     Example: 舷宽 → CVN-69 艾森豪威尔号
     Example: 推进功率 → CVN-68 尼米兹号

     INCLUDING CVN-68:
     舰总长 → CVN-68 尼米兹号 is CORRECT.
     舷宽 → CVN-68 尼米兹号 is CORRECT.
     推进功率 → CVN-68 尼米兹号 is CORRECT.
     航速 → CVN-68 尼米兹号 is CORRECT.
     舰员编制 → CVN-68 尼米兹号 is CORRECT.
     舰载机数量 → CVN-68 尼米兹号 is CORRECT.
     飞行甲板面积 → CVN-68 尼米兹号 is CORRECT.
     飞行甲板宽 → CVN-68 尼米兹号 is CORRECT.
     吃水深度 → CVN-68 尼米兹号 is CORRECT.
     续航力 → CVN-68 尼米兹号 is CORRECT.
     满载排水量 → CVN-68 尼米兹号 is CORRECT.
     标准排水量 → CVN-68 尼米兹号 is CORRECT.
     推进装置 → CVN-68 尼米兹号 is CORRECT.
     舰岛位置 → CVN-68 尼米兹号 is CORRECT.
     母港 → CVN-68 尼米兹号 is CORRECT.
     CVN-68 尼米兹号 → 舰总长 is WRONG.
     CVN-68 尼米兹号 → 舷宽 is WRONG.
     CVN-68 尼米兹号 → 推进功率 is WRONG.
     CVN-68 尼米兹号 → 航速 is WRONG.
     CVN-68 尼米兹号 → 舰员编制 is WRONG.
     CVN-68 尼米兹号 → 舰载机数量 is WRONG.

     NEVER reverse for ANY ship.

   **Rule 4 - BUILT, SERVICE_STATUS_OF, ARMOR_PROTECTION_OF**:
     Source_Entity → Ship_Instance

     Example: 建造船厂 → CVN-68 尼米兹号
     Example: 服役状态 → CVN-69 艾森豪威尔号
     Example: 双层舰壳 → CVN-68 尼米兹号

     INCLUDING CVN-68:
     建造船厂 → CVN-68 尼米兹号 is CORRECT.
     服役状态 → CVN-68 尼米兹号 is CORRECT.
     双层舰壳 → CVN-68 尼米兹号 is CORRECT.
     凯夫拉装甲 → CVN-68 尼米兹号 is CORRECT.
     HY-80 高强度钢 → CVN-68 尼米兹号 is CORRECT.
     水密隔舱壁 → CVN-68 尼米兹号 is CORRECT.
     防火隔壁 → CVN-68 尼米兹号 is CORRECT.
     水密隔舱 → CVN-68 尼米兹号 is CORRECT.
     纵向防雷舱壁 → CVN-68 尼米兹号 is CORRECT.
     X 形吸能支撑结构 → CVN-68 尼米兹号 is CORRECT.
     CVN-68 尼米兹号 → 建造船厂 is WRONG.
     CVN-68 尼米兹号 → 服役状态 is WRONG.
     CVN-68 尼米兹号 → 双层舰壳 is WRONG.
     CVN-68 尼米兹号 → 凯夫拉装甲 is WRONG.

     NEVER reverse for ANY ship.

   **Rule 5 - EQUIPPED_WITH**:
     Ship_Instance → Configuration
     Example: CVN-68 尼米兹号 → CVN-68 雷达套件
     Example: CVN-68 尼米兹号 → CVN-68 武器套件
     Example: CVN-68 尼米兹号 → CVN-68 舰载机联队
     NEVER reverse.

   **Rule 6 - Equipment to Configuration relationships**:
     RADAR_OF, WEAPON_OF, COUNTERMEASURE_OF, COMBAT_SYSTEM_OF,
     COMMUNICATION_OF, DATA_LINK_OF, GUN_OF, AIRCRAFT_OF,
     POWERPLANT_OF, CATAPULT_OF, ARRESTING_GEAR_OF, ARMOR_OF
     must follow:
     Equipment_Entity → Configuration

     Example: AN/SPS-48C/E → CVN-68 雷达套件
     Example: Mk 29 → CVN-69 武器套件
     Example: A4W 压水核反应堆 → CVN-68 动力套件
     Example: F/A-18E/F → CVN-68 舰载机联队
     Example: 勃朗宁 M2 → CVN-68 舰载火炮套件

     INCLUDING CVN-68:
     AN/SPS-48C/E → CVN-68 雷达套件 is CORRECT.
     AN/SPS-49(V)5 → CVN-68 雷达套件 is CORRECT.
     AN/SPS-43A → CVN-68 雷达套件 is CORRECT.
     AN/SPS-67 → CVN-68 雷达套件 is CORRECT.
     AN/SPQ-9A → CVN-68 雷达套件 is CORRECT.
     AN/SPQ-9B → CVN-68 雷达套件 is CORRECT.
     AN/SPN-46 → CVN-68 雷达套件 is CORRECT.
     AN/SPN-43C → CVN-68 雷达套件 is CORRECT.
     AN/SPN-41 → CVN-68 雷达套件 is CORRECT.
     Mk 91 NSSM → CVN-68 雷达套件 is CORRECT.
     Mk 95 → CVN-68 雷达套件 is CORRECT.
     SPS-64(V)9 → CVN-68 雷达套件 is CORRECT.
     URN-25 → CVN-68 雷达套件 is CORRECT.
     Mk 29 → CVN-68 武器套件 is CORRECT.
     RIM-7 → CVN-68 武器套件 is CORRECT.
     Mk 49 → CVN-68 武器套件 is CORRECT.
     RIM-116 → CVN-68 武器套件 is CORRECT.
     Mk 15 → CVN-68 武器套件 is CORRECT.
     Mk 38 → CVN-68 舰载火炮套件 is CORRECT.
     勃朗宁 M2 → CVN-68 舰载火炮套件 is CORRECT.
     AN/SLQ-32(V)4 → CVN-68 电子战套件 is CORRECT.
     AN/WLR-1H → CVN-68 电子战套件 is CORRECT.
     Mk 36 SRBOC → CVN-68 电子战套件 is CORRECT.
     AN/SLQ-25 → CVN-68 电子战套件 is CORRECT.
     SLQ-29 → CVN-68 电子战套件 is CORRECT.
     ACDS → CVN-68 作战系统套件 is CORRECT.
     NTDS → CVN-68 作战系统套件 is CORRECT.
     SSDS Mk 2 → CVN-68 作战系统套件 is CORRECT.
     MK-23 TAS → CVN-68 作战系统套件 is CORRECT.
     SRR-1 → CVN-68 通信套件 is CORRECT.
     WSC-3 → CVN-68 通信套件 is CORRECT.
     WSC-6 → CVN-68 通信套件 is CORRECT.
     USC-38 → CVN-68 通信套件 is CORRECT.
     LINK-4A → CVN-68 数据链套件 is CORRECT.
     LINK-11 → CVN-68 数据链套件 is CORRECT.
     LINK-14 → CVN-68 数据链套件 is CORRECT.
     LINK-16 → CVN-68 数据链套件 is CORRECT.
     F/A-18E/F → CVN-68 舰载机联队 is CORRECT.
     F/A-18C/D → CVN-68 舰载机联队 is CORRECT.
     E-2C → CVN-68 舰载机联队 is CORRECT.
     EA-6B → CVN-68 舰载机联队 is CORRECT.
     EA-18G → CVN-68 舰载机联队 is CORRECT.
     S-3A/B → CVN-68 舰载机联队 is CORRECT.
     SH-60F → CVN-68 舰载机联队 is CORRECT.
     HH-60H → CVN-68 舰载机联队 is CORRECT.
     MH-60R → CVN-68 舰载机联队 is CORRECT.
     C-2 → CVN-68 舰载机联队 is CORRECT.
     A4W 压水核反应堆 → CVN-68 动力套件 is CORRECT.
     蒸汽涡轮发动机 → CVN-68 动力套件 is CORRECT.
     四轴双主舵 → CVN-68 动力套件 is CORRECT.
     汽轮发电机 → CVN-68 动力套件 is CORRECT.
     应急柴油发电机 → CVN-68 动力套件 is CORRECT.
     弹射器 → CVN-68 弹射器套件 is CORRECT.
     拦阻索 → CVN-68 拦阻装置套件 is CORRECT.
     拦阻网 → CVN-68 拦阻装置套件 is CORRECT.
     双层舰壳 → CVN-68 装甲防护套件 is CORRECT.
     X 形吸能支撑结构 → CVN-68 装甲防护套件 is CORRECT.
     HY-80 高强度钢 → CVN-68 装甲防护套件 is CORRECT.
     水密隔舱壁 → CVN-68 装甲防护套件 is CORRECT.
     防火隔壁 → CVN-68 装甲防护套件 is CORRECT.
     水密隔舱 → CVN-68 装甲防护套件 is CORRECT.
     纵向防雷舱壁 → CVN-68 装甲防护套件 is CORRECT.
     凯夫拉装甲 → CVN-68 装甲防护套件 is CORRECT.
     CVN-68 雷达套件 → AN/SPS-48C/E is WRONG.
     CVN-68 武器套件 → Mk 29 is WRONG.
     CVN-68 舰载火炮套件 → 勃朗宁 M2 is WRONG.
     CVN-68 舰载机联队 → F/A-18E/F is WRONG.
     CVN-68 动力套件 → A4W 压水核反应堆 is WRONG.

     NEVER reverse for ANY ship.

   **Rule 7 - Equipment to Function relationships**:
     HAS_RADAR_FUNCTION, HAS_COUNTERMEASURE_FUNCTION, HAS_COMBAT_FUNCTION,
     HAS_COMMUNICATION_FUNCTION, HAS_WEAPON_FUNCTION, HAS_AIRCRAFT_FUNCTION,
     HAS_SHIPBOARD_GUN_FUNCTION
     must follow:
     Equipment_Entity → Function_Entity
     Example: AN/SPS-48C/E → 对空搜索
     Example: Mk 15 → 近防系统
     Example: 勃朗宁 M2 → 重机枪
     Example: F/A-18E/F → 战斗攻击机
     NEVER reverse. Function_Entity → Equipment_Entity is WRONG.

   **Self-Check Before Output**:
   Before final output, check every relationship according to Rule 1 - Rule 7 above.

   1. The direction must be determined by the exact relationship type, not only by the `_OF` suffix.

   2. If a relationship expresses a valid fact from the input text but its source and target are reversed, output the same relationship in the schema-correct direction defined by Rule 1 - Rule 7.

      Example:
      Wrong: CVN-68 尼米兹号 → 推进功率
      Correct: 推进功率 → CVN-68 尼米兹号

      Example:
      Wrong: 对空搜索 → AN/SPS-48E
      Correct: AN/SPS-48E → 对空搜索

   3. Do NOT output both directions for the same relationship.

   4. Do NOT create any relationship type outside the predefined relationship list.

   5. Do NOT create any entity name outside the known entity dictionary.

   6. If an entity appears in the input text with extra descriptive words, first normalize it to the closest valid entity name in the known entity dictionary.
      Example:
      "勃朗宁 M2 重机枪" → "勃朗宁 M2"
      "Mk 38 遥控机炮" → "Mk 38"

   7. If a relationship is clearly supported by the current ship block and both entities can be normalized to known dictionary entities, keep the relationship and output it in the schema-correct direction.

   8. Only skip a relationship when it is not supported by the input text, or when its entities cannot be normalized to any known dictionary entity.

   **Completeness Check**:
   - After processing ALL ships, verify that every Ship_Instance explicitly appearing in the input has been extracted.
   - For each Ship_Instance, extract only the relationships explicitly supported by its own [SHIP] block.
   - If a relationship is clearly present in the current ship block, keep it and output it in the schema-correct direction.
   - Do NOT delete a valid relationship only because its direction was initially reversed; correct the direction instead.
   - Do NOT copy relationships from other ships.
   - Do NOT infer missing equipment, attributes, or functions from other ships.
   - Do NOT add any relationship that is not explicitly supported by the current ship's own text.

   Use ONLY the relationship types provided.

3. **CRITICAL FORMAT RULES**:
The markers <entity_name> and <source_entity> in the format template below are ONLY PLACEHOLDERS.
You MUST replace them with the actual entity names without any quotes or angle brackets.

*   **WRONG**: ("entity"{tuple_delimiter}<CVN-68 尼米兹号>{tuple_delimiter}Ship_Instance{tuple_delimiter}无)
*   **WRONG**: ("entity"{tuple_delimiter}"CVN-68 尼米兹号"{tuple_delimiter}Ship_Instance{tuple_delimiter}无)
*   **CORRECT**: ("entity"{tuple_delimiter}CVN-68 尼米兹号{tuple_delimiter}Ship_Instance{tuple_delimiter}无)

Format each entity exactly as:
("entity"{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description)

Format each relationship exactly as:
("relationship"{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_type{tuple_delimiter}strength)

**Remember**: The final output must contain only the plain text names, never "<" or ">".

**Output Integrity Check**:
- Every entity_name in a relationship MUST be a complete, valid entity name that exists in the known entity dictionary.
- If an entity name is incomplete or truncated (e.g., "CVN(" or "AN("), DO NOT output it.
- If you cannot find the complete entity name, skip that relationship.


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
