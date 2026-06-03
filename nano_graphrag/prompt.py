"""
Prompts for nano-graphrag adapted to ship-class prototype classification.

任务目标：
1. 不再围绕 CVN-68 ~ CVN-77 具体舰实例抽取。
2. 只围绕 class_data.txt 中的已知舰级原型抽取。
3. 不允许出现未知类舰级名称；未知类只在后续开放集判断阶段通过规则/阈值输出。
4. nano-graphrag 只作为知识图谱/证据解释层，不作为最终分类器。
5. 字段名统一作为 Feature_Slot；字段取值统一构造成 “Slot=Value” 形式的值实体，避免所有卡槽共享泛化的“有/无”节点。
"""

GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}

PROMPTS[
    "claim_extraction"
] = """-Target activity-
You are an intelligent assistant that helps a human analyst analyze claims against entities presented in a text document.

-Goal-
Given a text document, an entity specification, and a claim description, extract all entities matching the specification and all claims against those entities.

-Steps-
1. Extract named entities matching the predefined entity specification.
2. For each entity, extract claims matching the claim description.
For each claim, extract:
- Subject
- Object
- Claim Type
- Claim Status: TRUE, FALSE, or SUSPECTED
- Claim Description
- Claim Date
- Claim Source Text

Format each claim as:
(<subject_entity>{tuple_delimiter}<object_entity>{tuple_delimiter}<claim_type>{tuple_delimiter}<claim_status>{tuple_delimiter}<claim_start_date>{tuple_delimiter}<claim_end_date>{tuple_delimiter}<claim_description>{tuple_delimiter}<claim_source>)

3. Return output as a single list using {record_delimiter}.
4. When finished, output {completion_delimiter}.

-Real Data-
Entity specification: {entity_specs}
Claim description: {claim_description}
Text: {input_text}
Output: """


PROMPTS[
    "community_report"
] = """你是一个舰船类别知识图谱分析专家。

请根据输入的实体和关系，生成“舰级原型特征总结”。注意：这里的对象是 Ship_Class，不是具体舷号或具体舰实例。

你必须严格输出 JSON，格式如下：

{{
  "known_class_profiles": [
    {{
      "ship_class": "舰级名称",
      "ship_category": "舰船大类",
      "key_visual_features": ["关键外观结构特征，使用 Slot=Value 形式"],
      "key_aviation_features": ["关键航空作业特征，使用 Slot=Value 形式"],
      "key_amphibious_features": ["关键两栖/登陆特征，使用 Slot=Value 形式"],
      "key_weapon_sensor_features": ["关键武器/传感器特征，使用 Slot=Value 形式"],
      "key_text_attributes": ["关键非视觉技术参数，使用 Slot=Value 形式"],
      "negative_features": ["用于排除误判的负特征，使用 Slot=Value 形式"],
      "classification_notes": "该舰级最适合通过哪些特征识别"
    }}
  ],
  "category_summary": [
    {{
      "ship_category": "舰船大类",
      "known_classes": ["该大类下的已知舰级"],
      "category_discriminative_features": ["该大类的主要判别特征"]
    }}
  ]
}}

关键要求：
1. 只能总结输入图谱中真实存在的 Ship_Class、Ship_Category 和特征，不要补充外部知识。
2. 不要输出任何未知类舰级名称。
3. 不要输出具体舷号、CVN-xx 或单舰实例。
4. 特征值应优先使用图谱中的 “Slot=Value” 节点名称，例如 Catapult=有、Well_Deck=无、Hull_Form=三体船。
5. 只输出 JSON，不要包含其他解释文字。

Text:
```
{input_text}
```

Output:
"""


PROMPTS["entity_extraction"] = """-Goal-
Given a structured class_data document about known warship classes, extract entities and relationships according to the schema below.

The document describes known ship-class prototypes for hierarchical classification:
Ship_Category -> Ship_Class -> Feature_Slot / Slot=Value.

This is NOT a concrete ship-instance extraction task.
Do NOT extract hull numbers or individual ship instances.
Do NOT extract unknown-class names.
Do NOT create entity names or relationship types outside the schema.

-Input Format-
The text is organized as:
- 【舰级名称】 marks one known ship-class profile.
- [CLASS] contains Ship_Category, Ship_Class, Known_Status, Known_Class.
- [VISUAL_STRUCTURE] contains basic visible structural features.
- [AVIATION_FEATURES] contains flight deck, helicopter spots, aircraft elevators, catapults, arresting gear and hangar features.
- [AMPHIBIOUS_FEATURES] contains well deck, stern gate, landing craft, vehicle deck and troop transport features.
- [TEXT_ATTRIBUTES] contains non-visual technical parameters.
- [WEAPON_SENSOR_FEATURES] contains VLS, main gun, CIWS, radar, sonar and missile feature abstractions.
- [EQUIPMENT_DETAILS] contains concrete equipment model names, if present.
- [MISSION_FEATURES] contains mission and combat capability descriptions.
- [NEGATIVE_FEATURES] contains explicitly absent capabilities or structures.
- [TEXT_STRONG_CUES] contains strong textual keywords.
Each section contains lines in "Slot: Value" format.

-Entity Types-
Use only the following entity types:
1. Ship_Category
2. Ship_Class
3. Known_Status
4. Feature_Group
5. Feature_Slot
6. Feature_Value
7. Equipment_Value
8. Mission_Value
9. Negative_Feature

-Allowed Ship_Category values-
航空母舰, 巡洋舰, 驱逐舰, 护卫舰, 两栖舰, 登陆舰

-Allowed Known Ship_Class values-
尼米兹级航空母舰
提康德罗加级导弹巡洋舰
阿利·伯克级驱逐舰
独立级濒海战斗舰
黄蜂级两栖攻击舰
圣安东尼奥级两栖船坞运输舰
惠德比岛级船坞登陆舰

-Forbidden unknown evaluation classes-
The following names are evaluation-only unknown classes. They must NOT be extracted, generated, or inserted into the graph:
金刚级驱逐舰, 村雨级护卫舰, 大邱级巡防舰

-Allowed Feature_Slot names-
The following field names are Feature_Slot entities. They are slot names, not ordinary values:

VISUAL_STRUCTURE:
Hull_Form, Bow_Form, Stern_Form, Superstructure_Position, Superstructure_Type,
Island_Presence, Island_Position, Bridge_Position,
Funnel_Presence, Funnel_Count, Funnel_Form, Funnel_Position,
Mast_Feature, Stealth_Shape, Freeboard_Level

AVIATION_FEATURES:
Flight_Deck_Type, Flight_Deck_Position, Helicopter_Spot_Count,
Aircraft_Elevator, Aircraft_Elevator_Count,
Catapult, Catapult_Count,
Arresting_Gear, Arresting_Gear_Count,
Hangar, Aircraft_Capacity_Level,
Fixed_Wing_Aircraft_Operation, STOVL_Aircraft_Operation

AMPHIBIOUS_FEATURES:
Well_Deck, Stern_Gate, Landing_Craft_Capability, Vehicle_Deck,
Troop_Transport, Amphibious_Assault_Capability, Landing_Craft_Capacity

TEXT_ATTRIBUTES:
Length_Overall, Beam, Draft, Standard_Displacement, Full_Load_Displacement,
Speed, Range, Crew, Aircraft_Capacity, Vehicle_Capacity, Troop_Capacity,
Landing_Craft_Capacity, Power_Output, Propulsion, Powerplant

WEAPON_SENSOR_FEATURES:
VLS_Presence, VLS_Count_Level, VLS_Position,
Main_Gun_Presence, Main_Gun_Position, Main_Gun_Caliber,
CIWS_Presence, Phased_Array_Radar, Radar_Array_Type,
Anti_Ship_Missile_Launcher, Sonar_Feature

EQUIPMENT_DETAILS:
Radar_System, Combat_System, Weapon_System, Countermeasure_System,
Communication_System, Data_Link, Aircraft, Powerplant_Detail,
Landing_Craft, Mission_Module

MISSION_FEATURES:
Primary_Mission, Air_Operation_Capability, Area_Air_Defense,
Anti_Submarine, Anti_Surface, Mine_Countermeasure,
Amphibious_Assault, Landing_Operation, Command_Control,
Fleet_Core, Patrol_Littoral

NEGATIVE_FEATURES:
No_Well_Deck, No_Stern_Gate, No_Catapult, No_Arresting_Gear,
No_Full_Flight_Deck, No_Large_VLS_As_Main_Feature,
No_Landing_Craft_Capability, No_Large_Aviation_Facility,
No_Fixed_Wing_Carrier_Operation, No_VLS

TEXT_STRONG_CUES:
Keywords

-CRITICAL Slot=Value Entity Naming Rule-
For every "Slot: Value" line, the value entity name MUST be constructed as:

Slot=ValueItem

Examples:
- Hull_Form: 三体船
  Feature_Slot entity: Hull_Form
  Feature_Value entity: Hull_Form=三体船

- Aircraft_Elevator: 有
  Feature_Slot entity: Aircraft_Elevator
  Feature_Value entity: Aircraft_Elevator=有

- Catapult: 无
  Feature_Slot entity: Catapult
  Feature_Value entity: Catapult=无

- Well_Deck: 有
  Feature_Slot entity: Well_Deck
  Feature_Value entity: Well_Deck=有

- Main_Gun_Position: 舰艏
  Feature_Slot entity: Main_Gun_Position
  Feature_Value entity: Main_Gun_Position=舰艏

- Radar_System: AN/SPY-1, AN/SPS-49
  Feature_Slot entity: Radar_System
  Equipment_Value entities: Radar_System=AN/SPY-1 and Radar_System=AN/SPS-49

Why this rule is mandatory:
- Do NOT create generic value nodes such as 有, 无, 0, 1, 强, 弱 alone.
- Do NOT store the value only in the Feature_Slot description.
- Do NOT use only an edge description to represent the value.
- The graph must explicitly contain Slot=Value nodes so that different slots do not share the same ambiguous "有/无" node.

-Allowed Relationship Types and Directions-
Use only these relationship types and directions:

1. CLASS_IN_CATEGORY
   Ship_Class -> Ship_Category
   Example: 尼米兹级航空母舰 -> 航空母舰

2. HAS_KNOWN_STATUS
   Ship_Class -> Known_Status
   Example: 尼米兹级航空母舰 -> Known

3. HAS_VISUAL_FEATURE
   Ship_Class -> Feature_Value
   Used for [VISUAL_STRUCTURE].

4. HAS_AVIATION_FEATURE
   Ship_Class -> Feature_Value
   Used for [AVIATION_FEATURES].

5. HAS_AMPHIBIOUS_FEATURE
   Ship_Class -> Feature_Value
   Used for [AMPHIBIOUS_FEATURES].

6. HAS_TEXT_ATTRIBUTE
   Ship_Class -> Feature_Value
   Used for [TEXT_ATTRIBUTES] and [TEXT_STRONG_CUES].

7. HAS_WEAPON_SENSOR_FEATURE
   Ship_Class -> Feature_Value
   Used for [WEAPON_SENSOR_FEATURES].

8. HAS_EQUIPMENT_DETAIL
   Ship_Class -> Equipment_Value
   Used for [EQUIPMENT_DETAILS].

9. HAS_MISSION_FEATURE
   Ship_Class -> Mission_Value
   Used for [MISSION_FEATURES].

10. HAS_NEGATIVE_FEATURE
    Ship_Class -> Negative_Feature
    Used for [NEGATIVE_FEATURES].

11. VALUE_OF_SLOT
    Feature_Value / Equipment_Value / Mission_Value / Negative_Feature -> Feature_Slot
    Example: Catapult=有 -> Catapult
    Example: Well_Deck=无 -> Well_Deck
    Example: Hull_Form=三体船 -> Hull_Form

12. SUPPORTS_CATEGORY
    Feature_Value / Equipment_Value / Mission_Value / Negative_Feature -> Ship_Category
    Only create this relationship when the category support is explicit or strongly implied by the slot/value.
    Examples:
    Catapult=有 -> 航空母舰
    Arresting_Gear=有 -> 航空母舰
    Well_Deck=有 -> 两栖舰
    Stern_Gate=有 -> 登陆舰
    Hull_Form=三体船 -> 护卫舰
    VLS_Presence=有 -> 巡洋舰
    VLS_Presence=有 -> 驱逐舰

13. BELONGS_TO_GROUP
    Feature_Slot -> Feature_Group
    Example: Flight_Deck_Type -> AVIATION_FEATURES
    Example: Hull_Form -> VISUAL_STRUCTURE

-Section to Relationship Mapping-
[VISUAL_STRUCTURE]       -> HAS_VISUAL_FEATURE
[AVIATION_FEATURES]      -> HAS_AVIATION_FEATURE
[AMPHIBIOUS_FEATURES]    -> HAS_AMPHIBIOUS_FEATURE
[TEXT_ATTRIBUTES]        -> HAS_TEXT_ATTRIBUTE
[WEAPON_SENSOR_FEATURES] -> HAS_WEAPON_SENSOR_FEATURE
[EQUIPMENT_DETAILS]      -> HAS_EQUIPMENT_DETAIL
[MISSION_FEATURES]       -> HAS_MISSION_FEATURE
[NEGATIVE_FEATURES]      -> HAS_NEGATIVE_FEATURE
[TEXT_STRONG_CUES]       -> HAS_TEXT_ATTRIBUTE

-Feature Group Meaning-
VISUAL_STRUCTURE: hull form, bow/stern form, superstructure, island, bridge, funnel, mast, stealth shape and freeboard.
AVIATION_FEATURES: flight deck, helicopter spots, aircraft elevator, catapult, arresting gear, hangar and aircraft operation.
AMPHIBIOUS_FEATURES: well deck, stern gate, landing craft, vehicle deck, troop transport and amphibious assault.
TEXT_ATTRIBUTES: length, beam, draft, displacement, speed, range, crew, capacity, propulsion and powerplant.
WEAPON_SENSOR_FEATURES: VLS, main gun, CIWS, phased-array radar, radar array, anti-ship missile and sonar.
EQUIPMENT_DETAILS: concrete radar, weapon, combat system, aircraft, landing craft, powerplant or mission-module names.
MISSION_FEATURES: primary mission and combat capability.
NEGATIVE_FEATURES: explicitly absent structures or capabilities.

-Extraction Rules-
1. For [CLASS]:
   - Extract Ship_Class from "Ship_Class:".
   - Extract Ship_Category from "Ship_Category:".
   - Extract Known_Status from "Known_Status:".
   - Create:
     Ship_Class -> Ship_Category with CLASS_IN_CATEGORY.
     Ship_Class -> Known_Status with HAS_KNOWN_STATUS.
   - Do NOT create a separate entity for Known_Class. Known_Class is only a human-readable field.

2. For each "Slot: Value" line in feature sections:
   - Slot must be a Feature_Slot entity exactly as written in the slot list above.
   - Value may contain multiple items separated by comma, Chinese comma, semicolon, or clearly listed alternatives.
   - If Value clearly contains multiple meaningful items, create one Slot=ValueItem entity for each meaningful item.
   - If a slash expresses synonymous or alternative wording in one phrase, keep it as one value unless it clearly lists separate features.
   - Keep the original Chinese/alphanumeric casing. Do NOT uppercase entity names.
   - The entity_name of the value entity MUST be Slot=ValueItem.
   - The entity_description should include: group name, slot name, raw value text.
   - Create:
     Ship_Class -> Slot=ValueItem with the section's relationship type.
     Slot=ValueItem -> Slot with VALUE_OF_SLOT.
     Slot -> Feature_Group with BELONGS_TO_GROUP.

3. For [EQUIPMENT_DETAILS]:
   - Slot is still a Feature_Slot.
   - Each concrete equipment model becomes Equipment_Value with entity_name = Slot=EquipmentName.
   - Do not split model names like AN/SPY-1, AN/SPS-49(V)5, Mk 41 VLS, RIM-116, LM2500, MV-22.
   - Do not create equipment values from generic phrases alone such as "雷达系统", "武器系统", "舰载机" unless the text gives a concrete name.
   - Example:
     Radar_System: AN/SPY-1
     entity: Radar_System=AN/SPY-1, entity_type: Equipment_Value
     relationship: Radar_System=AN/SPY-1 -> Radar_System, VALUE_OF_SLOT

4. For [MISSION_FEATURES]:
   - Each value becomes Mission_Value with entity_name = Slot=ValueItem.
   - Example:
     Primary_Mission: 舰载机航空作战
     entity: Primary_Mission=舰载机航空作战, entity_type: Mission_Value

5. For [NEGATIVE_FEATURES]:
   - Each value becomes Negative_Feature with entity_name = Slot=ValueItem.
   - Example:
     No_Catapult: 是
     entity: No_Catapult=是, entity_type: Negative_Feature
   - Negative features are important and must not be omitted.

6. For "未知", "无", "不适用", "非主要特征", "非主要能力":
   - These values may still be extracted when they represent an important absence or classification clue.
   - They must still follow Slot=Value format, such as Well_Deck=无, Catapult=无, VLS_Presence=非主要特征.
   - Do NOT create generic nodes "未知", "无", "不适用", "非主要特征".

7. Do not infer missing facts:
   - Extract only what is present in the current class profile.
   - Do not copy features from other classes.
   - Do not use external military knowledge.
   - Do not add unknown-class information.

8. Direction self-check:
   - Ship_Class must be source for CLASS_IN_CATEGORY and HAS_* relationships.
   - Slot=Value entities must point to Feature_Slot with VALUE_OF_SLOT.
   - Feature_Slot must point to Feature_Group with BELONGS_TO_GROUP.
   - Do not output both directions.
   - Do not create relationship types outside the allowed list.

-Output Format Rules-
Format each entity exactly as:
("entity"{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description)

Format each relationship exactly as:
("relationship"{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_type{tuple_delimiter}strength)

-Important Formatting Requirements-
1. The output must contain only plain text entity names.
2. Do not include angle brackets.
3. Do not include quotation marks around entity names.
4. Do not output any relationship type outside the allowed list.
5. Return output as a single list using {record_delimiter}.
6. When finished, output {completion_delimiter}.

-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""


PROMPTS[
    "summarize_entity_descriptions"
] = """你是一个舰船知识图谱摘要助手。

请根据给定实体名和描述列表，生成一个简洁、统一、无重复的中文摘要。
要求：
1. 保留实体名。
2. 如果实体名是 Slot=Value 形式，必须保留完整实体名，不要简化成 Value。
3. 不要改变型号大小写，例如 AN/SPY-1、Mk 41 VLS、LM2500。
4. 如果描述冲突，请说明存在不同资料表述，不要强行编造唯一事实。
5. 不要补充输入描述之外的信息。

#######
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""


PROMPTS[
    "entiti_continue_extraction"
] = """上一轮可能遗漏了 class_data 中的舰级、属性卡槽、Slot=Value 属性值或关系。请继续按同一格式补充遗漏内容。
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """请判断是否仍有 class_data 中明确出现的舰级、属性卡槽、Slot=Value 属性值或关系未被抽取。只回答 YES 或 NO。
"""

PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "Ship_Category",
    "Ship_Class",
    "Known_Status",
    "Feature_Group",
    "Feature_Slot",
    "Feature_Value",
    "Equipment_Value",
    "Mission_Value",
    "Negative_Feature",
]

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"


PROMPTS[
    "local_rag_response"
] = """---Role---

你是一个舰船层级识别解释助手。你只能基于给定知识图谱上下文回答，不要使用外部知识。

---Goal---

根据检索到的已知舰级原型知识，解释输入描述可能属于哪个舰船大类、是否匹配某个已知舰级，或者是否更适合判断为类别内未知类。

注意：
1. 你不是最终分类器；最终分类应由程序的 hierarchical_class_match 得分和开放集阈值决定。
2. 你只负责提供图谱证据解释。
3. 不要输出未知类舰级名称。
4. 不要编造图谱中没有的舰级或特征。
5. 特征证据请优先使用图谱中的 Slot=Value 形式，例如 Hull_Form=三体船、Catapult=无、Well_Deck=有。

你必须严格按照以下 JSON Schema 输出：

{{
  "category_evidence": [
    {{
      "ship_category": "航空母舰/巡洋舰/驱逐舰/护卫舰/两栖舰/登陆舰",
      "supporting_features": ["支持该大类的 Slot=Value 图谱特征"],
      "conflicting_features": ["与该大类冲突的 Slot=Value 图谱特征"]
    }}
  ],
  "known_class_evidence": [
    {{
      "ship_class": "已知舰级名称",
      "known_status": "Known",
      "matched_features": ["匹配的 Slot=Value 特征"],
      "missing_or_conflicting_features": ["缺失或冲突的关键 Slot=Value 特征"],
      "explanation": "简要说明"
    }}
  ],
  "open_set_note": "如果输入能判断大类但不充分匹配任何已知舰级，请说明应交由开放集阈值判断为类别内未知类；不要命名未知舰级。",
  "summary": "一句话总结图谱证据"
}}

---Data tables---

{context_data}

---Goal---

只输出 JSON，不要包含其他解释文字。
"""


PROMPTS[
    "global_map_rag_points"
] = """---Role---

You are a helpful assistant responding to questions about data in the provided tables.

---Goal---

Generate key points that respond to the user's question using only the provided data tables.
If the data tables do not contain sufficient information, say so. Do not make anything up.

Each key point must have:
- Description
- Importance Score: integer from 0 to 100

Output JSON:
{{
    "points": [
        {{"description": "Description of point 1", "score": score_value}},
        {{"description": "Description of point 2", "score": score_value}}
    ]
}}

---Data tables---

{context_data}

---Goal---

Generate the JSON key points.
"""


PROMPTS[
    "global_reduce_rag_response"
] = """---Role---

You are a helpful assistant synthesizing multiple reports about a dataset.

---Goal---

Generate a response of the target length and format that answers the user's question by summarizing the provided reports.
If the reports do not contain sufficient information, say so. Do not make anything up.
Do not include information unsupported by the reports.

---Target response length and format---
{response_type}

---Analyst Reports---
{report_data}

---Goal---

Generate the final response in markdown.
"""


PROMPTS[
    "naive_rag_response"
] = """你是一个舰船知识问答助手。请只根据下面提供的知识回答问题；如果知识不足，请直接说明，不要编造。

{content_data}

---Target response length and format---
{response_type}
"""


PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["default_text_separator"] = [
    "\n\n",
    "\r\n\r\n",
    "\n",
    "\r\n",
    "。",
    "．",
    ".",
    "！",
    "!",
    "？",
    "?",
    "；",
    ";",
    "，",
    ",",
    " ",
    "\t",
    "\u3000",
    "\u200b",
]
