# -*- coding: utf-8 -*-
"""
schema_config.py

用途：
1. 统一约束 LLM 文本解析输出的 JSON 卡槽；
2. 统一约束 class_data -> GraphML 的规则构图关系；
3. 统一提供属性值归一化、同义词映射、打分权重；
4. 支撑层级分类匹配：舰船大类判断 -> 已知舰级匹配 -> 类别内未知判断。

重要原则：
- 本文件只包含“已知类”的先验类别和舰级，不包含任何未知类舰级名称。
- LLM 只负责从文本中抽取属性卡槽，不直接判断最终舰级。
- 最终分类由 hierarchical_class_match 等程序模块根据卡槽、图谱和阈值完成。
"""

import re
import unicodedata
from typing import Any, Dict, List, Tuple


# ============================================================
# 1. 舰船大类与已知舰级
# ============================================================

SHIP_CATEGORIES: List[str] = [
    "航空母舰",
    "巡洋舰",
    "驱逐舰",
    "护卫舰",
    "两栖舰",
    "登陆舰",
]


KNOWN_SHIP_CLASSES: Dict[str, List[str]] = {
    "航空母舰": ["尼米兹级航空母舰"],
    "巡洋舰": ["提康德罗加级导弹巡洋舰"],
    "驱逐舰": ["阿利·伯克级驱逐舰"],
    "护卫舰": ["独立级濒海战斗舰"],
    "两栖舰": [
        "黄蜂级两栖攻击舰",
        "圣安东尼奥级两栖船坞运输舰",
    ],
    "登陆舰": ["惠德比岛级船坞登陆舰"],
}

KNOWN_STATUS_VALUES: List[str] = ["Known"]
UNKNOWN_OUTPUT_TEMPLATE: str = "{category}类别内未知类"


# ============================================================
# 2. 固定输出卡槽 schema
# ============================================================
# direct_text_parse_v2() 必须严格按照这些字段输出。
# 未提到的信息统一输出“未知”；明确看不清/无法判断输出“不确定”；明确不存在输出“无”。

SLOT_SCHEMA: Dict[str, List[str]] = {
    "CLASS": [
        "Ship_Category",
        "Ship_Class",
        "Known_Status",
        "Known_Class",
    ],

    "VISUAL_STRUCTURE": [
        "Hull_Form",                 # 船体形式
        "Bow_Form",                  # 舰首形态
        "Stern_Form",                # 舰尾形态
        "Superstructure_Position",   # 上层建筑位置
        "Superstructure_Type",       # 上层建筑类型
        "Island_Presence",           # 是否有舰岛
        "Island_Position",           # 舰岛位置
        "Bridge_Position",           # 舰桥位置
        "Funnel_Presence",           # 是否有烟囱/排烟结构
        "Funnel_Count",              # 烟囱数量
        "Funnel_Form",               # 烟囱形态
        "Funnel_Position",           # 烟囱位置
        "Mast_Feature",              # 桅杆特征
        "Stealth_Shape",             # 隐身化外形
        "Freeboard_Level",           # 干舷高度
    ],

    "AVIATION_FEATURES": [
        "Flight_Deck_Type",          # 飞行甲板类型
        "Flight_Deck_Position",      # 飞行甲板位置
        "Helicopter_Spot_Count",     # 直升机起降点数量
        "Aircraft_Elevator",         # 是否有飞机升降机
        "Aircraft_Elevator_Count",   # 飞机升降机数量
        "Catapult",                  # 是否有弹射器
        "Catapult_Count",            # 弹射器数量
        "Arresting_Gear",            # 是否有拦阻装置
        "Arresting_Gear_Count",      # 拦阻索/拦阻装置数量
        "Hangar",                    # 机库
        "Aircraft_Capacity_Level",   # 航空搭载能力等级
        "Fixed_Wing_Aircraft_Operation",  # 固定翼飞机作业能力
        "STOVL_Aircraft_Operation",       # 短距起飞/垂直降落飞机能力
    ],

    "AMPHIBIOUS_FEATURES": [
        "Well_Deck",                     # 坞舱
        "Stern_Gate",                    # 艉门/舰尾开口
        "Landing_Craft_Capability",      # 登陆艇搭载/投放能力
        "Vehicle_Deck",                  # 车辆甲板
        "Troop_Transport",               # 运兵能力
        "Amphibious_Assault_Capability", # 两栖攻击能力
        "Landing_Craft_Capacity",        # 登陆艇搭载数量
    ],

    "TEXT_ATTRIBUTES": [
        "Length_Overall",            # 总长
        "Beam",                      # 舰宽
        "Draft",                     # 吃水
        "Standard_Displacement",     # 标准排水量
        "Full_Load_Displacement",    # 满载排水量
        "Speed",                     # 航速
        "Range",                     # 续航力
        "Crew",                      # 舰员
        "Aircraft_Capacity",         # 舰载机/直升机数量
        "Vehicle_Capacity",          # 车辆搭载能力
        "Troop_Capacity",            # 运兵数量
        "Landing_Craft_Capacity",    # 登陆艇搭载数量
        "Power_Output",              # 推进功率
        "Propulsion",                # 推进方式/推进构型
        "Powerplant",                # 动力系统/动力来源
    ],

    "WEAPON_SENSOR_FEATURES": [
        "VLS_Presence",                 # 是否有垂直发射系统
        "VLS_Count_Level",              # 垂发数量等级
        "VLS_Position",                 # 垂发位置
        "Main_Gun_Presence",            # 是否有主炮
        "Main_Gun_Position",            # 主炮位置
        "Main_Gun_Caliber",             # 主炮口径
        "CIWS_Presence",                # 近防系统
        "Phased_Array_Radar",           # 相控阵雷达
        "Radar_Array_Type",             # 雷达阵面类型
        "Anti_Ship_Missile_Launcher",   # 反舰导弹发射装置
        "Sonar_Feature",                # 声呐特征
    ],

    "EQUIPMENT_DETAILS": [
        "Radar_System",             # 具体雷达型号
        "Combat_System",            # 具体作战系统
        "Weapon_System",            # 具体武器型号
        "Countermeasure_System",    # 电子战/干扰系统
        "Communication_System",     # 通信系统
        "Data_Link",                # 数据链
        "Aircraft",                 # 舰载机/直升机型号
        "Powerplant_Detail",        # 具体动力型号
        "Landing_Craft",            # 登陆艇型号
        "Mission_Module",           # 任务模块，主要用于独立级濒海战斗舰
    ],

    "MISSION_FEATURES": [
        "Primary_Mission",              # 主要任务
        "Air_Operation_Capability",     # 航空作业能力
        "Area_Air_Defense",             # 区域防空能力
        "Anti_Submarine",               # 反潜能力
        "Anti_Surface",                 # 对海/反舰能力
        "Mine_Countermeasure",          # 反水雷能力
        "Amphibious_Assault",           # 两栖攻击能力
        "Landing_Operation",            # 登陆作战能力
        "Command_Control",              # 指挥控制能力
        "Fleet_Core",                   # 是否舰队核心/编队核心
        "Patrol_Littoral",              # 濒海巡逻/近海作战能力
    ],

    "NEGATIVE_FEATURES": [
        "No_Well_Deck",                       # 无坞舱
        "No_Stern_Gate",                      # 无艉门
        "No_Catapult",                        # 无弹射器
        "No_Arresting_Gear",                  # 无拦阻装置
        "No_Full_Flight_Deck",                # 无全通飞行甲板
        "No_Large_VLS_As_Main_Feature",        # 垂发不是主要特征
        "No_Landing_Craft_Capability",         # 无登陆艇能力
        "No_Large_Aviation_Facility",          # 无大型航空设施
        "No_Fixed_Wing_Carrier_Operation",     # 无传统固定翼航母作业能力
        "No_VLS",                              # 无垂直发射系统
    ],

    "TEXT_STRONG_CUES": [
        "Keywords",
    ],
}

# 反向索引：slot -> group
SLOT_TO_GROUP: Dict[str, str] = {
    slot: group
    for group, slots in SLOT_SCHEMA.items()
    for slot in slots
}

ALL_SLOTS: List[str] = [
    slot
    for group in SLOT_SCHEMA
    for slot in SLOT_SCHEMA[group]
]


# ============================================================
# 3. 每组特征在 GraphML 中对应的关系类型
# ============================================================

GROUP_TO_RELATION: Dict[str, str] = {
    "VISUAL_STRUCTURE": "HAS_VISUAL_FEATURE",
    "AVIATION_FEATURES": "HAS_AVIATION_FEATURE",
    "AMPHIBIOUS_FEATURES": "HAS_AMPHIBIOUS_FEATURE",
    "TEXT_ATTRIBUTES": "HAS_TEXT_ATTRIBUTE",
    "WEAPON_SENSOR_FEATURES": "HAS_WEAPON_SENSOR_FEATURE",
    "EQUIPMENT_DETAILS": "HAS_EQUIPMENT_DETAIL",
    "MISSION_FEATURES": "HAS_MISSION_FEATURE",
    "NEGATIVE_FEATURES": "HAS_NEGATIVE_FEATURE",
    "TEXT_STRONG_CUES": "HAS_TEXT_ATTRIBUTE",
}

RELATION_SCHEMA: Dict[str, Tuple[str, str]] = {
    "CLASS_IN_CATEGORY": ("Ship_Class", "Ship_Category"),
    "HAS_KNOWN_STATUS": ("Ship_Class", "Known_Status"),
    "HAS_VISUAL_FEATURE": ("Ship_Class", "Feature_Value"),
    "HAS_AVIATION_FEATURE": ("Ship_Class", "Feature_Value"),
    "HAS_AMPHIBIOUS_FEATURE": ("Ship_Class", "Feature_Value"),
    "HAS_TEXT_ATTRIBUTE": ("Ship_Class", "Feature_Value"),
    "HAS_WEAPON_SENSOR_FEATURE": ("Ship_Class", "Feature_Value"),
    "HAS_EQUIPMENT_DETAIL": ("Ship_Class", "Equipment_Value"),
    "HAS_MISSION_FEATURE": ("Ship_Class", "Mission_Value"),
    "HAS_NEGATIVE_FEATURE": ("Ship_Class", "Negative_Feature"),
    "VALUE_OF_SLOT": ("Feature_Value", "Feature_Slot"),
    "SUPPORTS_CATEGORY": ("Feature_Value", "Ship_Category"),
}


# ============================================================
# 4. 标准属性值词典
# ============================================================
# 这些值用于约束 LLM 输出。不是所有字段都必须完全限定，
# 数值字段、具体装备字段可以保留原文或归一化后的短语。

COMMON_UNKNOWN_VALUES: List[str] = ["未知", "不确定", "未提及", "无"]

VALUE_VOCAB: Dict[str, List[str]] = {
    # ---------- VISUAL_STRUCTURE ----------
    "Hull_Form": [
        "大型单体船", "单体船", "三体船", "宽体三体船", "双体船", "不确定", "未知"
    ],
    "Bow_Form": [
        "球鼻艏", "尖艏", "大型舰艏", "尖削舰艏", "外飘舰艏", "登陆舰舰艏", "不确定", "未知"
    ],
    "Stern_Form": [
        "宽大平直舰尾", "常规方艉", "平直舰尾", "船坞艉门", "大型船坞艉门", "开放式艉门", "宽大三体舰艉", "不确定", "未知"
    ],
    "Superstructure_Position": [
        "右舷", "舰体中部偏后", "中前部", "前部", "前部/中前部", "中部", "不确定", "未知"
    ],
    "Superstructure_Type": [
        "舰岛式上层建筑",
        "传统多层上层建筑",
        "隐身化封闭式上层建筑",
        "大型箱形隐身化上层建筑",
        "低矮隐身化上层建筑",
        "传统多层两栖舰上层建筑",
        "不确定", "未知"
    ],
    "Island_Presence": ["有", "无", "不确定", "未知"],
    "Island_Position": ["右舷舰岛", "左舷舰岛", "无舰岛", "不确定", "未知"],
    "Bridge_Position": [
        "右舷舰岛内",
        "前部上层建筑内",
        "中前部上层建筑内",
        "前部/中前部上层建筑内",
        "前部/中前部大型上层建筑内",
        "不确定", "未知"
    ],
    "Funnel_Presence": ["有", "无", "不明显", "有但不明显", "不确定", "未知"],
    "Funnel_Count": ["0", "1", "2", "多个", "不适用", "不确定", "未知"],
    "Funnel_Form": [
        "独立烟囱", "双烟囱", "纵列式双烟囱", "与上层建筑融合", "与舰岛融合", "低可见排烟结构", "不适用", "不确定", "未知"
    ],
    "Funnel_Position": [
        "舰岛区域", "中部", "后部", "中部至后部上层建筑区域", "上层建筑内", "两侧", "不适用", "不确定", "未知"
    ],
    "Mast_Feature": [
        "舰岛集成桅杆", "独立桅杆", "高桅杆", "雷达桅杆", "集成式雷达桅杆", "三角桅杆", "封闭式综合桅杆", "常规桅杆", "不确定", "未知"
    ],
    "Stealth_Shape": [
        "普通大型航母外形", "传统大型水面舰外形", "强隐身化外形", "强隐身化箱形外形", "低矮隐身化外形", "普通两栖舰外形", "不明显", "不确定", "未知"
    ],
    "Freeboard_Level": ["高", "中高", "中", "低", "不确定", "未知"],

    # ---------- AVIATION_FEATURES ----------
    "Flight_Deck_Type": [
        "全通飞行甲板", "斜角飞行甲板", "艉部直升机甲板", "大型艉部直升机飞行甲板", "无明显飞行甲板", "不确定", "未知"
    ],
    "Flight_Deck_Position": [
        "全舰贯通并延伸至舰艉", "全舰贯通", "舰尾", "中后部", "无", "不确定", "未知"
    ],
    "Helicopter_Spot_Count": ["0", "1", "1-2", "2", "多个", "不确定", "未知"],
    "Aircraft_Elevator": ["有", "无", "无航母式飞机升降机", "不确定", "未知"],
    "Aircraft_Elevator_Count": ["0", "2座", "4座", "数量未知", "不适用", "不确定", "未知"],
    "Catapult": ["有", "无", "不确定", "未知"],
    "Catapult_Count": ["0", "2台", "4台", "数量未知", "不确定", "未知"],
    "Arresting_Gear": ["有", "无", "不确定", "未知"],
    "Arresting_Gear_Count": ["0", "3条", "4条", "数量未知", "不确定", "未知"],
    "Hangar": ["大型机库", "直升机机库", "有", "无", "无或有限", "不确定", "未知"],
    "Aircraft_Capacity_Level": [
        "大量固定翼舰载机", "30架级", "少量直升机", "1-2架直升机级", "2架直升机或直升机+无人机级", "有限航空能力", "无", "不确定", "未知"
    ],
    "Fixed_Wing_Aircraft_Operation": ["有", "无", "STOVL为主", "非主要能力", "不确定", "未知"],
    "STOVL_Aircraft_Operation": ["有", "无", "非主要能力", "不确定", "未知"],

    # ---------- AMPHIBIOUS_FEATURES ----------
    "Well_Deck": ["有", "无", "大型坞舱", "全通式泛水坞舱", "不确定", "未知"],
    "Stern_Gate": ["有", "无", "有艉部任务舱门", "不确定", "未知"],
    "Landing_Craft_Capability": ["强", "有", "无", "LCAC/LCM登陆艇", "LCAC/LCU/两栖战车", "小艇/无人艇投放回收能力", "不确定", "未知"],
    "Vehicle_Deck": ["有", "无", "任务舱/模块舱", "车辆甲板", "不确定", "未知"],
    "Troop_Transport": ["强", "有", "无", "弱/非主要能力", "不确定", "未知"],
    "Amphibious_Assault_Capability": ["强", "中强", "弱", "弱/非主要能力", "无", "不确定", "未知"],
    "Landing_Craft_Capacity": ["3艘LCAC", "4艘LCAC", "2艘LCAC或1艘LCU", "无", "不确定", "未知"],

    # ---------- WEAPON_SENSOR_FEATURES ----------
    "VLS_Presence": ["有", "无", "可选", "通常无", "无或非主要特征", "非主要特征", "不确定", "未知"],
    "VLS_Count_Level": ["高", "中", "低", "无", "90-96单元级", "122单元级", "不确定", "未知"],
    "VLS_Position": ["舰艏", "舰艉", "舰艏和舰艉", "前后均有", "不作为主要识别特征", "无", "不确定", "未知"],
    "Main_Gun_Presence": ["有", "无", "有小口径自卫炮位", "无大型舰艏主炮", "不确定", "未知"],
    "Main_Gun_Position": ["舰艏", "舰艉", "舰艏和舰艉", "无大型舰艏主炮", "小口径自卫炮位", "无", "不确定", "未知"],
    "Main_Gun_Caliber": ["127mm级", "76mm级", "57mm级", "25mm级", "小口径自卫武器", "无", "不确定", "未知"],
    "CIWS_Presence": ["有", "无", "有/部分构型不同", "不确定", "未知"],
    "Phased_Array_Radar": ["有", "无", "非主要特征", "部分后期舰可能有", "不确定", "未知"],
    "Radar_Array_Type": ["四面固定相控阵", "AN/SPY-1四面固定相控阵", "AN/SPY-1D四面固定相控阵", "旋转搜索雷达", "常规搜索/导航雷达", "搜索雷达/综合桅杆", "多型搜索/空管/火控雷达", "不确定", "未知"],
    "Anti_Ship_Missile_Launcher": ["有", "无", "可根据任务模块加装", "非主要特征", "不确定", "未知"],
    "Sonar_Feature": ["舰首声呐", "拖曳阵列声呐", "舰首声呐和拖曳阵列声呐", "任务模块化反潜/反水雷设备", "非主要特征", "无", "不确定", "未知"],
}


# ============================================================
# 5. 全局同义词与标准化映射
# ============================================================
# 这部分用于把用户口语、百科表达、图像描述表达统一为标准表达。
# 这里先放高频规则，后续测试时继续补充。

GLOBAL_VALUE_ALIASES: Dict[str, str] = {
    # 飞行甲板
    "全舰甲板": "全通飞行甲板",
    "贯通甲板": "全通飞行甲板",
    "直通甲板": "全通飞行甲板",
    "一整条飞行甲板": "全通飞行甲板",
    "平直大甲板": "全通飞行甲板",
    "斜向甲板": "斜角飞行甲板",
    "斜直两段式飞行甲板": "斜角飞行甲板",
    "斜角甲板": "斜角飞行甲板",
    "舰尾停机坪": "艉部直升机甲板",
    "尾部停机坪": "艉部直升机甲板",
    "船尾直升机平台": "艉部直升机甲板",
    "舰尾直升机平台": "艉部直升机甲板",

    # 舰岛 / 舰桥 / 上层建筑
    "右边有舰岛": "右舷舰岛",
    "右侧舰岛": "右舷舰岛",
    "舰岛在右边": "右舷舰岛",
    "右舷岛式建筑": "右舷舰岛",
    "驾驶室在右侧舰岛": "右舷舰岛内",
    "舰桥在右侧舰岛": "右舷舰岛内",
    "舰桥在前部": "前部上层建筑内",
    "驾驶室在前部": "前部上层建筑内",
    "舰桥在船头附近": "前部上层建筑内",
    "驾驶室在船头附近": "前部上层建筑内",

    # 烟囱
    "双烟囱": "双烟囱",
    "两个烟囱": "双烟囱",
    "两个排烟口": "双烟囱",
    "烟囱和上层建筑融合": "与上层建筑融合",
    "烟囱不明显": "低可见排烟结构",
    "看不到明显烟囱": "低可见排烟结构",

    # 艉门 / 坞舱 / 登陆艇
    "船尾开口": "艉门",
    "舰尾开口": "艉门",
    "尾门": "艉门",
    "船尾能打开": "艉门",
    "舰尾能打开": "艉门",
    "泛水坞舱": "坞舱",
    "井围甲板": "坞舱",
    "船坞舱": "坞舱",
    "气垫登陆艇": "LCAC",

    # 武器/传感器
    "垂发": "垂直发射系统",
    "垂直发射井": "垂直发射系统",
    "导弹发射井": "垂直发射系统",
    "船头有炮": "舰艏主炮",
    "舰艏大炮": "舰艏主炮",
    "前甲板主炮": "舰艏主炮",
    "四面雷达阵面": "四面固定相控阵",
    "四面相控阵": "四面固定相控阵",
    "固定雷达阵面": "四面固定相控阵",

    # 船体
    "三体舰": "三体船",
    "三体船体": "三体船",
    "三个船体": "三体船",
    "中间一个主船体两侧两个辅船体": "三体船",
}

# 针对具体 slot 的映射，优先级高于 GLOBAL_VALUE_ALIASES。
SLOT_VALUE_ALIASES: Dict[str, Dict[str, str]] = {
    "Funnel_Presence": {
        "双烟囱": "有",
        "两个烟囱": "有",
        "与上层建筑融合": "有但不明显",
        "烟囱和上层建筑融合": "有但不明显",
        "不明显": "不明显",
        "看不到明显烟囱": "不明显",
        "没有烟囱": "无",
        "无烟囱": "无",
    },
    "Funnel_Count": {
        "双烟囱": "2",
        "两个烟囱": "2",
        "两个排烟口": "2",
        "没有烟囱": "0",
        "无烟囱": "0",
    },
    "Island_Presence": {
        "右舷舰岛": "有",
        "左舷舰岛": "有",
        "有舰岛": "有",
        "无舰岛": "无",
        "没有舰岛": "无",
    },
    "Catapult": {
        "弹射器": "有",
        "蒸汽弹射器": "有",
        "没有弹射器": "无",
        "无弹射器": "无",
    },
    "Arresting_Gear": {
        "拦阻索": "有",
        "拦阻装置": "有",
        "拦阻系统": "有",
        "没有拦阻索": "无",
        "无拦阻索": "无",
    },
    "Well_Deck": {
        "坞舱": "有",
        "泛水坞舱": "有",
        "井围甲板": "有",
        "船坞舱": "有",
        "没有坞舱": "无",
        "无坞舱": "无",
    },
    "Stern_Gate": {
        "艉门": "有",
        "舰尾开口": "有",
        "船尾开口": "有",
        "尾门": "有",
        "没有艉门": "无",
        "无艉门": "无",
    },
    "Main_Gun_Presence": {
        "舰艏主炮": "有",
        "船头有炮": "有",
        "主炮": "有",
        "没有主炮": "无",
        "无主炮": "无",
    },
    "VLS_Presence": {
        "垂发": "有",
        "垂直发射系统": "有",
        "垂直发射井": "有",
        "没有垂发": "无",
        "无垂发": "无",
    },
}


# ============================================================
# 6. 具体装备型号别名
# ============================================================
# 这些主要服务于百科文本输入。注意：这不是核心分类依据，只作为强证据补充。

EQUIPMENT_ALIASES: Dict[str, str] = {
    # 通用写法
    "MK41": "Mk 41 VLS",
    "Mk41": "Mk 41 VLS",
    "MK-41": "Mk 41 VLS",
    "MK 41": "Mk 41 VLS",
    "垂直发射系统": "Mk 41 VLS",

    "MK45": "Mk 45舰炮",
    "Mk45": "Mk 45舰炮",
    "MK-45": "Mk 45舰炮",
    "127毫米舰炮": "Mk 45舰炮",

    "MK15": "Mk 15 CIWS",
    "Mk15": "Mk 15 CIWS",
    "密集阵": "Mk 15 CIWS",
    "密集阵近防炮": "Mk 15 CIWS",

    "SPY-1": "AN/SPY-1",
    "AN/SPY1": "AN/SPY-1",
    "SPY-1D": "AN/SPY-1D",
    "AN/SPY1D": "AN/SPY-1D",

    "LM2500": "LM2500燃气轮机",
    "A4W": "A4W压水核反应堆",

    "LCAC气垫登陆艇": "LCAC",
    "气垫登陆艇": "LCAC",
    "LCU登陆艇": "LCU",

    "宙斯盾": "宙斯盾战斗系统",
    "宙斯盾系统": "宙斯盾战斗系统",
}


# ============================================================
# 7. 大类支持规则
# ============================================================
# 用于第一层大类判断。某些属性值天然支持某个大类。
# 权重可在实验中继续调整。

CATEGORY_FEATURE_HINTS: Dict[str, Dict[str, float]] = {
    "航空母舰": {
        "全通飞行甲板": 2.0,
        "斜角飞行甲板": 2.5,
        "弹射器": 3.0,
        "拦阻索": 3.0,
        "大量固定翼舰载机": 2.5,
        "右舷舰岛": 1.5,
        "核动力航空母舰": 2.0,
    },
    "巡洋舰": {
        "导弹巡洋舰": 3.0,
        "宙斯盾巡洋舰": 3.0,
        "122单元级": 2.5,
        "舰艏和舰艉各1门": 1.5,
        "区域防空": 2.0,
        "指挥控制": 1.5,
    },
    "驱逐舰": {
        "导弹驱逐舰": 3.0,
        "宙斯盾驱逐舰": 3.0,
        "舰艏主炮": 1.5,
        "90-96单元级": 2.0,
        "四面固定相控阵": 2.0,
        "多用途导弹驱逐舰": 2.5,
    },
    "护卫舰": {
        "濒海战斗舰": 3.0,
        "三体船": 3.0,
        "大型艉部直升机飞行甲板": 2.0,
        "高速浅吃水": 2.0,
        "模块化任务": 2.0,
        "喷水推进": 1.5,
    },
    "两栖舰": {
        "两栖攻击舰": 3.0,
        "两栖船坞运输舰": 3.0,
        "全通飞行甲板": 1.5,
        "坞舱": 3.0,
        "LCAC": 2.0,
        "海军陆战队": 2.0,
        "STOVL": 2.0,
        "MV-22": 1.5,
    },
    "登陆舰": {
        "船坞登陆舰": 3.0,
        "大型坞舱": 3.0,
        "大型船坞艉门": 2.5,
        "艉门": 2.0,
        "LCAC": 2.0,
        "登陆艇": 2.0,
    },
}


# ============================================================
# 8. 匹配权重
# ============================================================
# 用于第二层已知舰级匹配。
# 原则：越具有区分度的字段，权重越高。

SLOT_WEIGHTS: Dict[str, float] = {
    # 大类和身份
    "Ship_Category": 3.0,
    "Ship_Class": 3.0,

    # 视觉结构
    "Hull_Form": 2.5,
    "Bow_Form": 1.0,
    "Stern_Form": 1.5,
    "Superstructure_Position": 1.0,
    "Superstructure_Type": 2.0,
    "Island_Presence": 1.5,
    "Island_Position": 1.5,
    "Bridge_Position": 1.0,
    "Funnel_Presence": 0.8,
    "Funnel_Count": 1.0,
    "Funnel_Form": 1.0,
    "Funnel_Position": 0.8,
    "Mast_Feature": 1.0,
    "Stealth_Shape": 1.5,
    "Freeboard_Level": 0.8,

    # 航空能力
    "Flight_Deck_Type": 2.5,
    "Flight_Deck_Position": 2.0,
    "Helicopter_Spot_Count": 1.2,
    "Aircraft_Elevator": 1.5,
    "Aircraft_Elevator_Count": 1.5,
    "Catapult": 3.0,
    "Catapult_Count": 2.0,
    "Arresting_Gear": 3.0,
    "Arresting_Gear_Count": 2.0,
    "Hangar": 1.5,
    "Aircraft_Capacity_Level": 1.5,
    "Fixed_Wing_Aircraft_Operation": 2.0,
    "STOVL_Aircraft_Operation": 1.5,

    # 两栖/登陆能力
    "Well_Deck": 3.0,
    "Stern_Gate": 2.5,
    "Landing_Craft_Capability": 2.5,
    "Vehicle_Deck": 1.5,
    "Troop_Transport": 1.5,
    "Amphibious_Assault_Capability": 2.0,
    "Landing_Craft_Capacity": 2.0,

    # 非视觉技术参数
    "Length_Overall": 1.2,
    "Beam": 1.0,
    "Draft": 0.8,
    "Standard_Displacement": 1.2,
    "Full_Load_Displacement": 1.5,
    "Speed": 1.0,
    "Range": 0.8,
    "Crew": 0.8,
    "Aircraft_Capacity": 1.2,
    "Vehicle_Capacity": 1.2,
    "Troop_Capacity": 1.2,
    "Power_Output": 0.8,
    "Propulsion": 1.0,
    "Powerplant": 1.0,

    # 武器/传感器
    "VLS_Presence": 2.0,
    "VLS_Count_Level": 2.0,
    "VLS_Position": 1.5,
    "Main_Gun_Presence": 1.5,
    "Main_Gun_Position": 1.5,
    "Main_Gun_Caliber": 1.2,
    "CIWS_Presence": 0.8,
    "Phased_Array_Radar": 2.0,
    "Radar_Array_Type": 2.0,
    "Anti_Ship_Missile_Launcher": 1.0,
    "Sonar_Feature": 1.0,

    # 任务能力
    "Primary_Mission": 2.0,
    "Air_Operation_Capability": 1.5,
    "Area_Air_Defense": 1.5,
    "Anti_Submarine": 1.0,
    "Anti_Surface": 1.0,
    "Mine_Countermeasure": 1.5,
    "Amphibious_Assault": 2.0,
    "Landing_Operation": 2.0,
    "Command_Control": 1.0,
    "Fleet_Core": 1.5,
    "Patrol_Littoral": 1.5,

    # 文本强提示
    "Keywords": 2.0,
}

# 负特征冲突权重：用户明确观察到某特征，但某已知类明确没有，则扣分。
NEGATIVE_CONFLICT_WEIGHTS: Dict[str, float] = {
    "No_Well_Deck": 3.0,
    "No_Stern_Gate": 2.5,
    "No_Catapult": 3.0,
    "No_Arresting_Gear": 3.0,
    "No_Full_Flight_Deck": 2.5,
    "No_Large_VLS_As_Main_Feature": 1.5,
    "No_Landing_Craft_Capability": 2.5,
    "No_Large_Aviation_Facility": 2.0,
    "No_Fixed_Wing_Carrier_Operation": 2.5,
    "No_VLS": 2.0,
}


# ============================================================
# 9. 开放集判断阈值
# ============================================================
# 这些阈值是初始值，后续需要根据测试集调参。

CATEGORY_CONFIDENCE_THRESHOLD: float = 0.45
KNOWN_CLASS_CONFIDENCE_THRESHOLD: float = 0.60
KNOWN_CLASS_MARGIN_THRESHOLD: float = 0.08

# 大类分数高，但已知舰级最高分低于该阈值时，输出“类别内未知类”。
OPEN_SET_CLASS_THRESHOLD: float = 0.55


# ============================================================
# 10. 标准化工具函数
# ============================================================

def normalize_basic_text(value: Any) -> str:
    """基础字符串归一化：全半角、空格、标点。"""
    if value is None:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("，", ",")
    text = text.replace("；", ";")
    text = text.replace("、", ",")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_slot_value(slot: str, value: Any) -> str:
    """
    对单个 slot 的值做标准化。
    优先使用 slot 级别别名，再使用全局别名，最后返回基础归一化文本。
    """
    text = normalize_basic_text(value)
    if not text:
        return "未知"

    # slot 专属映射优先
    slot_aliases = SLOT_VALUE_ALIASES.get(slot, {})
    if text in slot_aliases:
        return slot_aliases[text]

    # 全局映射
    if text in GLOBAL_VALUE_ALIASES:
        return GLOBAL_VALUE_ALIASES[text]

    # 装备细节映射
    if slot in SLOT_SCHEMA.get("EQUIPMENT_DETAILS", []) and text in EQUIPMENT_ALIASES:
        return EQUIPMENT_ALIASES[text]

    return text


def normalize_equipment_name(value: Any) -> str:
    """具体装备型号归一化。"""
    text = normalize_basic_text(value)
    return EQUIPMENT_ALIASES.get(text, text)


def get_slot_group(slot: str) -> str:
    """返回某个 slot 所属的 feature group。"""
    return SLOT_TO_GROUP.get(slot, "")


def get_group_relation(group: str) -> str:
    """返回某个 feature group 在 GraphML 中对应的关系类型。"""
    return GROUP_TO_RELATION.get(group, "")


def is_known_class(ship_class: str) -> bool:
    """判断舰级是否属于已知类。"""
    for classes in KNOWN_SHIP_CLASSES.values():
        if ship_class in classes:
            return True
    return False


def get_category_of_known_class(ship_class: str) -> str:
    """返回已知舰级所属大类。"""
    for category, classes in KNOWN_SHIP_CLASSES.items():
        if ship_class in classes:
            return category
    return ""


def empty_observed_schema() -> Dict[str, Dict[str, str]]:
    """
    生成一个空的 observed JSON 结构。
    direct_text_parse_v2() 的输出应与该结构一致。
    """
    result: Dict[str, Dict[str, str]] = {}
    for group, slots in SLOT_SCHEMA.items():
        if group == "CLASS":
            continue
        result[group] = {slot: "未知" for slot in slots}
    return result


def validate_observed_schema(observed: Dict[str, Any]) -> List[str]:
    """
    校验 LLM 输出是否符合 SLOT_SCHEMA。
    返回错误列表；空列表表示通过。
    """
    errors: List[str] = []

    if not isinstance(observed, dict):
        return ["observed must be a dict"]

    for group, slots in SLOT_SCHEMA.items():
        if group == "CLASS":
            continue

        if group not in observed:
            errors.append(f"missing group: {group}")
            continue

        if not isinstance(observed[group], dict):
            errors.append(f"group {group} must be a dict")
            continue

        for slot in slots:
            if slot not in observed[group]:
                errors.append(f"missing slot: {group}.{slot}")

        for slot in observed[group].keys():
            if slot not in slots:
                errors.append(f"unexpected slot: {group}.{slot}")

    return errors


if __name__ == "__main__":
    print("SHIP_CATEGORIES:", SHIP_CATEGORIES)
    print("KNOWN_SHIP_CLASSES:", KNOWN_SHIP_CLASSES)
    print("TOTAL_SLOTS:", len(ALL_SLOTS))

# ============================================================
# v14 扩展：真实未知类开放集测试所需的同义词/标准值补充
# ============================================================
# 注意：这里仍然不加入任何未知舰级名称，只补充结构、装备、任务和参数表达。

# 1) 视觉/航空/武器/任务标准值补充
VALUE_VOCAB.setdefault("Mast_Feature", [])
for _v in ["重型四角格子桅", "四角桁架桅杆", "四角格子桅杆", "轻质十字桅杆"]:
    if _v not in VALUE_VOCAB["Mast_Feature"]:
        VALUE_VOCAB["Mast_Feature"].insert(0, _v)

VALUE_VOCAB.setdefault("Hangar", [])
for _v in ["有直升机机库", "无机库但有直升机平台", "无直升机机库"]:
    if _v not in VALUE_VOCAB["Hangar"]:
        VALUE_VOCAB["Hangar"].insert(0, _v)

VALUE_VOCAB.setdefault("VLS_Count_Level", [])
for _v in ["16单元级", "90单元级", "90具级", "96单元级", "122单元级"]:
    if _v not in VALUE_VOCAB["VLS_Count_Level"]:
        VALUE_VOCAB["VLS_Count_Level"].insert(0, _v)

VALUE_VOCAB.setdefault("VLS_Position", [])
for _v in ["中部", "舰桥前方", "前后均有", "舰艏和舰艉"]:
    if _v not in VALUE_VOCAB["VLS_Position"]:
        VALUE_VOCAB["VLS_Position"].insert(0, _v)

VALUE_VOCAB.setdefault("Primary_Mission", [])
for _v in ["反潜为主", "区域防空", "弹道导弹防御", "近海巡逻", "巡防护航", "濒海模块化任务"]:
    if _v not in VALUE_VOCAB["Primary_Mission"]:
        VALUE_VOCAB["Primary_Mission"].insert(0, _v)

VALUE_VOCAB.setdefault("Propulsion", [])
for _v in ["CODLOG", "COGAG", "双轴推进", "双轴CPP双舵"]:
    if _v not in VALUE_VOCAB["Propulsion"]:
        VALUE_VOCAB["Propulsion"].insert(0, _v)

# 2) 全局同义表达补充
GLOBAL_VALUE_ALIASES.update({
    # 桅杆/外形
    "重型四角格子桅杆": "重型四角格子桅",
    "传统式重型四角格子桅": "重型四角格子桅",
    "四角格子桅杆": "重型四角格子桅",
    "四角桁架": "四角桁架桅杆",
    "轻质十字桅": "轻质十字桅杆",

    # 机库/直升机平台
    "没有机库但有直升机平台": "无机库但有直升机平台",
    "无机库但有直升机平台": "无机库但有直升机平台",
    "无直升机库": "无直升机机库",
    "直升机库": "有直升机机库",

    # 船体
    "普通单体船": "单体船",
    "传统单体船": "单体船",
    "常规单体船": "单体船",

    # 垂发数量和类型
    "16单元垂直发射系统": "16单元级",
    "16单元垂发": "16单元级",
    "90具Mk41": "90单元级",
    "90具MK41": "90单元级",
    "90个Mk41": "90单元级",
    "96单元": "96单元级",
    "122单元": "122单元级",
    "122枚": "122单元级",
    "K-VLS": "K-VLS",
    "韩国垂发": "K-VLS",
    "Mk48": "Mk 48 VLS",
    "MK48": "Mk 48 VLS",

    # 任务/类别表达
    "巡防舰": "护卫舰",
    "反潜护卫舰": "护卫舰",
    "通用护卫舰": "护卫舰",
    "反潜为主": "反潜为主",
    "弹道导弹防御": "弹道导弹防御",
    "近海巡逻": "近海巡逻",
    "巡防护航": "巡防护航",

    # 两栖/登陆
    "井围甲板": "坞舱",
    "大型井围甲板": "大型坞舱",
    "大型泛水坞舱": "大型坞舱",
    "4艘LCAC": "4艘LCAC",
    "四艘LCAC": "4艘LCAC",
})

# 3) slot 级别映射补充
SLOT_VALUE_ALIASES.setdefault("Hull_Form", {}).update({
    "普通单体船": "单体船",
    "传统单体船": "单体船",
    "常规单体船": "单体船",
})
SLOT_VALUE_ALIASES.setdefault("Mast_Feature", {}).update({
    "重型四角格子桅杆": "重型四角格子桅",
    "传统式重型四角格子桅": "重型四角格子桅",
    "四角格子桅杆": "重型四角格子桅",
    "四角桁架": "四角桁架桅杆",
    "轻质十字桅": "轻质十字桅杆",
})
SLOT_VALUE_ALIASES.setdefault("Hangar", {}).update({
    "没有机库但有直升机平台": "无机库但有直升机平台",
    "无机库但有直升机平台": "无机库但有直升机平台",
    "无直升机库": "无直升机机库",
    "直升机库": "有直升机机库",
})
SLOT_VALUE_ALIASES.setdefault("VLS_Count_Level", {}).update({
    "16单元垂直发射系统": "16单元级",
    "16单元垂发": "16单元级",
    "90具Mk41": "90单元级",
    "90具MK41": "90单元级",
    "96单元": "96单元级",
    "122单元": "122单元级",
    "122枚": "122单元级",
})
SLOT_VALUE_ALIASES.setdefault("Propulsion", {}).update({
    "柴电燃联合推进": "CODLOG",
    "柴电燃交替动力方式": "CODLOG",
    "复合机械/电力推进": "CODLOG",
    "复合燃气涡轮与燃气涡轮推进": "COGAG",
})
SLOT_VALUE_ALIASES.setdefault("Primary_Mission", {}).update({
    "以反潜任务为主": "反潜为主",
    "反潜任务为主": "反潜为主",
    "主要承担区域防空": "区域防空",
    "弹道导弹防御": "弹道导弹防御",
})

# 4) 具体装备型号补充
EQUIPMENT_ALIASES.update({
    "OPS-24": "OPS-24 3D对空搜索雷达",
    "OPS24": "OPS-24 3D对空搜索雷达",
    "OPS-28D": "OPS-28D平面搜索雷达",
    "OPS28D": "OPS-28D平面搜索雷达",
    "OYQ-8": "OYQ-8作战系统",
    "OYQ8": "OYQ-8作战系统",
    "OYQ-9": "OYQ-9作战系统",
    "OYQ9": "OYQ-9作战系统",
    "K-VLS": "K-VLS垂直发射系统",
    "KVLS": "K-VLS垂直发射系统",
    "Mk48": "Mk 48 VLS",
    "MK48": "Mk 48 VLS",
    "Mk 48": "Mk 48 VLS",
    "CODLOG": "CODLOG柴电燃联合推进",
    "COGAG": "COGAG燃气轮机联合推进",
    "NOLQ-2": "NOLQ-2电战系统",
    "NOLQ2": "NOLQ-2电战系统",
    "NOLQ-2/3": "NOLQ-2/3电战系统",
    "SPS-550K": "SPS-550K 3D对空搜索雷达",
    "SPS550K": "SPS-550K 3D对空搜索雷达",
    "SQS-240": "SQS-240舰首声纳",
    "海星": "海星反舰导弹",
    "K-SAAM": "K-SAAM防空导弹",
    "红鲨": "红鲨反潜火箭",
    "蓝鲨": "蓝鲨反潜导弹",
    "奥托梅莱拉127mm": "奥托·梅莱拉127mm舰炮",
    "奥托·梅莱拉127mm": "奥托·梅莱拉127mm舰炮",
    "奥托布雷达76毫米": "奥托·梅莱拉76mm舰炮",
    "OTO Melara": "奥托·梅莱拉舰炮",
})

