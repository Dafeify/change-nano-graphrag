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
# 11. v26 known-only 已知类签名规则
# ============================================================
# 严格原则：
# 1) 只包含 7 个已知舰级自身的强锚点、支持特征和已知类之间的冲突特征；
# 2) 不包含任何未知舰级名称；
# 3) 不包含从未知类样本中提取的专属装备、参数、结构签名；
# 4) open-set 由“未充分匹配任何已知舰级”触发，而不是由“预先知道未知类长什么样”触发。

SIGNATURE_RULE_THRESHOLDS: Dict[str, float] = {
    # 已知类保护阈值：anchor / support / adjusted 任一组合足够时，允许从 open_set 拉回已知类。
    "known_anchor_protect": 10.0,
    "known_support_protect": 12.0,
    "known_protect": 12.0,

    # 高分强制纠偏：必须有 anchor 或 support，不能仅靠共享强特征。
    "known_force_correct": 18.0,

    # 冲突阈值：仅使用其他已知类强特征作为冲突，不使用未知类专属特征。
    "known_conflict_block": 12.0,
    "known_only_open_set_conflict": 14.0,

    # 弱已知类拒识：没有 anchor/support，且综合分低，才输出类别内未知，避免过度 open-set。
    "weak_match_anchor_max": 3.0,
    "weak_match_support_max": 5.0,
    "weak_match_adjusted_max": 8.0,

    # 单已知舰级大类保护：该大类只有一个已知舰级时，支持证据达到该阈值且无冲突即可补全。
    "single_known_anchor_fill": 6.0,
    "single_known_support_fill": 8.0,
    "single_known_adjusted_fill": 10.0,
}

# 唯一锚点：更能确认某个已知舰级的特征。共享特征不要放太高。
KNOWN_CLASS_ANCHOR_SIGNATURES: Dict[str, List[Tuple[str, float]]] = {
    "尼米兹级航空母舰": [
        ("尼米兹级", 22), ("尼米兹", 20),
        ("核动力航空母舰", 18), ("超级航母", 14),
        ("10万吨", 14), ("十万吨", 14), ("满载排水量超过10万吨", 22), ("104200吨", 18),
        ("舰长332.8", 16), ("332.8米", 16), ("飞行甲板宽76.4", 16),
        ("4台蒸汽弹射器", 24), ("四台蒸汽弹射器", 24), ("4台弹射器", 22),
        ("4条拦阻索", 24), ("四条拦阻索", 24),
        ("A4W", 18), ("A4W压水反应堆", 22), ("C-13", 16),
        ("固定翼舰载机", 12), ("超过100架", 16), ("舰载机85", 12), ("85-90架", 12),
    ],
    "提康德罗加级导弹巡洋舰": [
        ("提康德罗加级", 22), ("提康德罗加", 20),
        ("导弹巡洋舰", 22), ("宙斯盾巡洋舰", 22), ("唯一一级巡洋舰", 18),
        ("第一种正式使用宙斯盾系统", 22),
        ("122单元", 30), ("共122单元", 32), ("122单元MK41", 34), ("122单元Mk41", 34),
        ("16组八联装", 30), ("16组八联装MK41", 34), ("16组八联装Mk41", 34),
        ("舰首和舰尾各有", 24), ("舰艏和舰艉各有", 24), ("双主炮", 22),
        ("航母战斗群的指挥中心", 20), ("指挥舰", 16), ("舰队指挥", 16), ("能当指挥舰", 18),
        ("比伯克级大", 18), ("比阿利·伯克级大", 18),
        ("舰长172.8", 16), ("172.8m", 16), ("172.8米", 16),
        ("9480吨", 16), ("9500吨", 14), ("满载排水量约9500吨", 16),
    ],
    "阿利·伯克级驱逐舰": [
        ("阿利·伯克级", 22), ("阿利伯克级", 22), ("伯克级", 18),
        ("美国海军主力驱逐舰", 18), ("美国最常见的那种宙斯盾驱逐舰", 18),
        ("SPY-1D", 24), ("AN/SPY-1D", 24),
        ("Flight IIA", 24), ("FlightIIA", 24), ("IIA型", 16),
        ("96单元", 24), ("96单元MK41", 28), ("96单元Mk41", 28), ("96管", 18),
        ("9238吨", 18), ("满载排水量约9200吨", 14), ("9200吨", 12),
        ("舰长155.29", 14), ("155.29m", 14),
        ("两座直升机库", 22), ("增设了直升机库", 18), ("后面有直升机库", 14),
        ("海鹰直升机", 12), ("SH-60", 12), ("MH-60", 12),
    ],
    "独立级濒海战斗舰": [
        ("独立级", 22), ("濒海战斗舰", 24),
        ("三体船", 28), ("三体结构", 28), ("三体船型", 28), ("三个船体", 26), ("三个船身", 26),
        ("中央主船体两侧", 20), ("左右支撑结构", 20),
        ("铝合金船体", 16), ("铝合金造", 14),
        ("任务模块", 16), ("模块化任务", 16), ("任务包", 14),
        ("航速45", 16), ("45-50节", 18), ("最高航速超过45节", 20),
        ("舰长127.6", 12), ("127.6米", 12), ("舰宽31.6", 12),
        ("57毫米", 14), ("57mm", 14),
    ],
    "黄蜂级两栖攻击舰": [
        ("黄蜂级", 22), ("两栖攻击舰", 24), ("LHD", 18), ("小航母", 14),
        ("全通飞行甲板", 10), ("全通式飞行甲板", 10),
        ("垂直起降战斗机", 22), ("AV-8B", 22), ("F-35B", 22), ("STOVL", 22),
        ("海军陆战队远征部队", 16), ("运输一整支海军陆战队", 14),
        ("舰长253.2", 14), ("253.2米", 14),
        ("满载排水量41150吨", 20), ("41150吨", 20), ("四万多吨", 12),
        ("3艘LCAC", 22), ("三艘LCAC", 22), ("坞舱长82.1", 14),
        ("未观察到弹射器和拦阻索", 16), ("没有弹射器和拦阻索", 16), ("无弹射器和拦阻索", 16),
    ],
    "圣安东尼奥级两栖船坞运输舰": [
        ("圣安东尼奥级", 22), ("圣安东尼奥", 20),
        ("两栖船坞运输舰", 28), ("船坞运输舰", 24), ("LPD", 18), ("两栖运输舰", 20),
        ("封闭式桅杆", 20), ("一体化桅杆", 20), ("先进的封闭式桅杆", 22),
        ("舰长208m", 14), ("舰长208米", 14), ("满载排水量25300吨", 20), ("25300吨", 18),
        ("MV-22", 22), ("鱼鹰", 22), ("倾转旋翼机", 22),
        ("车辆甲板面积2230", 16), ("货舱容积962", 16),
        ("720名海军陆战队员", 16),
        ("2艘LCAC", 22), ("两艘LCAC", 22), ("坞舱可容纳2艘LCAC", 24),
        ("有坞舱和直升机库", 18), ("能运兵和车辆", 14),
    ],
    "惠德比岛级船坞登陆舰": [
        ("惠德比岛级", 22), ("惠德比", 20),
        ("船坞登陆舰", 28), ("船坞登陆", 24), ("LSD", 18),
        ("大型井围甲板", 26), ("井围甲板", 24), ("大型泛水坞舱", 22), ("大坞舱", 18),
        ("专为搭载LCAC", 24), ("专为LCAC设计", 24), ("专为搭载LCAC气垫登陆艇设计", 26),
        ("4艘LCAC", 26), ("四艘LCAC", 26), ("可容纳4艘LCAC", 26),
        ("21艘LCM-6", 16),
        ("舰长185.8", 14), ("185.8m", 14),
        ("满载排水量16100吨", 20), ("16100吨", 20), ("16000吨", 12), ("一万六千吨", 12),
        ("627名海军陆战队员", 14),
        ("无机库", 14), ("没有直升机库", 14), ("飞行甲板支持直升机起降但无机库", 22),
        ("运输登陆装备", 14),
    ],
}

# 支持特征：不一定唯一，但如果同一已知舰级命中多项，且无冲突，可用于避免过度 open-set。
KNOWN_CLASS_SUPPORT_SIGNATURES: Dict[str, List[Tuple[str, float]]] = {
    "尼米兹级航空母舰": [
        ("航空母舰", 6), ("航母", 6), ("大型航母", 8), ("主力航母", 8),
        ("核动力", 8), ("甲板特别宽", 6), ("很多飞机", 6), ("好多飞机", 6),
        ("弹射器", 8), ("拦阻索", 8), ("把飞机弹出去", 8), ("弹出去的装置", 8),
        ("右舷舰岛", 5), ("斜角", 6),
    ],
    "提康德罗加级导弹巡洋舰": [
        ("巡洋舰", 6), ("宙斯盾舰", 5), ("AN/SPY-1", 6), ("SPY-1", 5),
        ("很多垂直发射", 6), ("垂发数量很多", 6), ("前后都有垂发", 6),
        ("标准防空导弹", 5), ("战斧巡航导弹", 5), ("航母护航", 6),
        ("指挥中心", 6), ("指挥控制", 5), ("舰体修长", 4), ("上层建筑较高", 4),
        ("舰尾设有直升机库", 5), ("比伯克级大一点", 8),
    ],
    "阿利·伯克级驱逐舰": [
        ("驱逐舰", 5), ("导弹驱逐舰", 5), ("主力驱逐舰", 7),
        ("宙斯盾驱逐舰", 7), ("带宙斯盾", 6),
        ("四面相控阵雷达", 6), ("四个大平板雷达", 6),
        ("Mk41", 4), ("MK41", 4), ("战斧导弹", 5), ("标准导弹", 4),
        ("后面有直升机库", 7), ("大概90多个", 6),
    ],
    "独立级濒海战斗舰": [
        ("军舰", 2), ("速度很快", 5), ("跑得特别快", 6), ("外形很特别", 5),
        ("宽大的飞行甲板", 4), ("可同时操作两架直升机", 5),
        ("比较轻", 4), ("拉姆防空导弹", 4),
    ],
    "黄蜂级两栖攻击舰": [
        ("两栖舰", 4), ("全通甲板", 5), ("全通飞行甲板", 5), ("直升机", 3),
        ("坞舱", 5), ("登陆艇", 4), ("气垫艇", 5),
        ("能当小航母用", 8), ("后面还有坞舱", 7), ("大型坞舱门", 8),
        ("多个直升机起降点", 6),
    ],
    "圣安东尼奥级两栖船坞运输舰": [
        ("两栖舰", 4), ("美国的两栖运输舰", 10), ("有坞舱能放登陆艇", 8),
        ("也有直升机库", 6), ("直升机库", 4), ("能运兵和车辆", 8),
        ("运兵和装备", 8), ("车辆甲板", 6), ("货舱", 5), ("一体化的", 5),
    ],
    "惠德比岛级船坞登陆舰": [
        ("登陆舰", 5), ("大坞舱", 8), ("放好几艘气垫登陆艇", 8),
        ("用来放登陆艇", 7), ("大型车辆甲板", 5), ("直升机平台但没有机库", 8),
        ("未观察到直升机库结构", 8), ("运输登陆装备", 8),
    ],
}

# 强匹配 = anchor + support 的弱化版本，主要用于候选排序。
KNOWN_CLASS_STRONG_SIGNATURES: Dict[str, List[Tuple[str, float]]] = {}

def _v26_extend_sig(target: Dict[str, List[Tuple[str, float]]], cls: str, items: List[Tuple[str, float]]):
    bucket = target.setdefault(cls, [])
    existing = {str(x[0]) for x in bucket}
    for term, weight in items:
        if str(term) not in existing:
            bucket.append((term, float(weight)))
            existing.add(str(term))

for _cls, _items in KNOWN_CLASS_ANCHOR_SIGNATURES.items():
    _v26_extend_sig(KNOWN_CLASS_STRONG_SIGNATURES, _cls, [(t, max(6.0, w * 0.65)) for t, w in _items])
for _cls, _items in KNOWN_CLASS_SUPPORT_SIGNATURES.items():
    _v26_extend_sig(KNOWN_CLASS_STRONG_SIGNATURES, _cls, [(t, max(3.0, w * 0.8)) for t, w in _items])

# 冲突特征：只使用其他已知类的强特征作为冲突依据。
KNOWN_CLASS_CONFLICT_SIGNATURES: Dict[str, List[Tuple[str, float]]] = {
    "尼米兹级航空母舰": [
        ("没有弹射器", 14), ("无弹射器", 14), ("没有拦阻索", 14), ("无拦阻索", 14),
        ("两栖攻击舰", 14), ("垂直起降战斗机", 12), ("AV-8B", 12), ("F-35B", 12),
        ("两栖船坞运输舰", 16), ("船坞登陆舰", 16), ("井围甲板", 14), ("4艘LCAC", 14),
        ("三体船", 16), ("濒海战斗舰", 16),
    ],
    "提康德罗加级导弹巡洋舰": [
        ("阿利·伯克级", 20), ("伯克级", 14), ("Flight IIA", 14), ("96单元", 14),
        ("三体船", 16), ("濒海战斗舰", 16),
        ("全通飞行甲板", 10), ("两栖攻击舰", 14), ("船坞登陆舰", 16), ("两栖船坞运输舰", 16),
        ("10万吨", 16), ("核动力航空母舰", 16),
    ],
    "阿利·伯克级驱逐舰": [
        ("提康德罗加级", 20), ("导弹巡洋舰", 18), ("巡洋舰", 8),
        ("122单元", 24), ("16组八联装", 24), ("双主炮", 20), ("舰首和舰尾各有", 20),
        ("三体船", 16), ("濒海战斗舰", 16),
        ("全通飞行甲板", 12), ("两栖攻击舰", 14), ("船坞登陆舰", 16), ("两栖船坞运输舰", 16),
    ],
    "独立级濒海战斗舰": [
        ("不是三体船", 18), ("没有三体船", 18),
        ("导弹巡洋舰", 18), ("122单元", 18), ("宙斯盾巡洋舰", 18),
        ("阿利·伯克级", 18), ("Flight IIA", 16), ("96单元", 14),
        ("两栖攻击舰", 16), ("船坞登陆舰", 16), ("两栖船坞运输舰", 16),
        ("10万吨", 16), ("核动力航空母舰", 16),
    ],
    "黄蜂级两栖攻击舰": [
        ("弹射器", 12), ("拦阻索", 12), ("传统弹射型航母", 16), ("核动力航空母舰", 16), ("10万吨", 16),
        ("两栖船坞运输舰", 14), ("LPD", 14), ("圣安东尼奥", 18), ("2艘LCAC", 14),
        ("船坞登陆舰", 16), ("LSD", 16), ("井围甲板", 16), ("4艘LCAC", 18),
        ("导弹巡洋舰", 16), ("导弹驱逐舰", 12),
    ],
    "圣安东尼奥级两栖船坞运输舰": [
        ("全通飞行甲板", 12), ("两栖攻击舰", 18), ("LHD", 16), ("STOVL", 14), ("AV-8B", 14), ("F-35B", 14), ("3艘LCAC", 16),
        ("船坞登陆舰", 18), ("LSD", 16), ("井围甲板", 16), ("4艘LCAC", 18),
        ("导弹巡洋舰", 16), ("导弹驱逐舰", 12), ("核动力航空母舰", 16),
    ],
    "惠德比岛级船坞登陆舰": [
        ("两栖船坞运输舰", 18), ("LPD", 16), ("圣安东尼奥", 18),
        ("封闭式桅杆", 14), ("一体化桅杆", 14), ("2艘LCAC", 16), ("MV-22", 14), ("鱼鹰", 14),
        ("全通飞行甲板", 14), ("两栖攻击舰", 18), ("AV-8B", 14), ("F-35B", 14), ("3艘LCAC", 14),
        ("导弹巡洋舰", 16), ("导弹驱逐舰", 12), ("核动力航空母舰", 16),
    ],
}

# 大类通用提示：只帮助判断大类，不直接匹配任何未知舰级。
CATEGORY_FEATURE_HINTS.setdefault("护卫舰", {}).update({
    "护卫舰": 3.0, "巡防舰": 2.5, "反潜护卫舰": 3.0, "中型通用护卫舰": 2.5,
    "现代护卫舰": 2.0, "单体护卫舰": 2.0,
})
CATEGORY_FEATURE_HINTS.setdefault("驱逐舰", {}).update({
    "大型防空导弹驱逐舰": 3.0, "防空导弹驱逐舰": 3.0, "大型驱逐舰": 2.0, "反导驱逐舰": 2.0,
})
CATEGORY_FEATURE_HINTS.setdefault("巡洋舰", {}).update({
    "指挥舰": 2.0, "比伯克级大": 2.0, "比阿利·伯克级大": 2.0,
})
CATEGORY_FEATURE_HINTS.setdefault("两栖舰", {}).update({
    "两栖运输舰": 2.5, "两栖船坞运输舰": 3.0, "两栖船坞": 2.5,
    "运兵和车辆": 2.0, "运兵和装备": 2.0,
})
CATEGORY_FEATURE_HINTS.setdefault("登陆舰", {}).update({
    "船坞登陆舰": 3.0, "船坞登陆": 2.5, "大坞舱": 2.0, "登陆装备": 2.0,
})
CATEGORY_FEATURE_HINTS.setdefault("航空母舰", {}).update({
    "主力航母": 3.0, "大型航母": 2.5, "核动力航母": 3.0,
    "有弹射器": 2.5, "很多飞机": 2.0, "好多飞机": 2.0,
})



# ============================================================
# 12. v27 known-only balanced open-set 规则校准
# ============================================================
# 严格原则：
# - 仍然只使用 7 个已知舰级自身的 anchor / support / conflict；
# - 不加入任何未知舰级名称；
# - 不加入未知类专属装备、参数、结构签名；
# - 通过“已知类证据是否充足”决定是否拒识，而不是通过“像某个未知类”决定。

SIGNATURE_RULE_THRESHOLDS.update({
    "known_anchor_protect": 8.0,
    "known_support_protect": 8.0,
    "known_protect": 7.0,
    "known_force_correct": 28.0,
    "known_conflict_block": 18.0,
    "known_only_open_set_conflict": 22.0,
    "weak_match_anchor_max": 0.5,
    "weak_match_support_max": 4.5,
    "weak_match_adjusted_max": 6.0,
    "single_known_anchor_fill": 4.0,
    "single_known_support_fill": 6.0,
    "single_known_adjusted_fill": 6.5,
    "cross_category_anchor_force": 22.0,
    "cross_category_support_force": 18.0,
    "cross_category_adjusted_force": 16.0,
})


def _v27_remove_sig(table: Dict[str, List[Tuple[str, float]]], cls: str, terms: List[str]):
    remove = set(terms)
    if cls in table:
        table[cls] = [(t, w) for (t, w) in table[cls] if str(t) not in remove]


def _v27_add_sig(table: Dict[str, List[Tuple[str, float]]], cls: str, items: List[Tuple[str, float]]):
    bucket = table.setdefault(cls, [])
    existing = {str(t) for t, _ in bucket}
    for t, w in items:
        if str(t) not in existing:
            bucket.append((t, float(w)))
            existing.add(str(t))

# 1) 修正容易造成跨大类误判的“共享特征”。
# 黄蜂级不能只靠“全通飞行甲板”作为锚点，否则容易吸走尼米兹级；应由两栖攻击、STOVL、AV-8B、坞舱/LCAC等共同确认。
_v27_remove_sig(KNOWN_CLASS_ANCHOR_SIGNATURES, "黄蜂级两栖攻击舰", ["全通飞行甲板", "全通式飞行甲板"])
_v27_add_sig(KNOWN_CLASS_SUPPORT_SIGNATURES, "黄蜂级两栖攻击舰", [
    ("全通飞行甲板", 4), ("全通式飞行甲板", 4), ("小航母", 8),
    ("坞舱门", 8), ("大型坞舱门", 10), ("多个直升机起降点", 6),
])

# 阿利·伯克级不能只靠泛泛的“宙斯盾驱逐舰 / Mk41 / 127mm”闭集硬判，避免吸走未知驱逐舰。
_v27_remove_sig(KNOWN_CLASS_SUPPORT_SIGNATURES, "阿利·伯克级驱逐舰", ["驱逐舰", "导弹驱逐舰", "宙斯盾驱逐舰", "带宙斯盾", "Mk41", "MK41"])
_v27_add_sig(KNOWN_CLASS_SUPPORT_SIGNATURES, "阿利·伯克级驱逐舰", [
    ("驱逐舰", 3), ("导弹驱逐舰", 3), ("宙斯盾驱逐舰", 4), ("带宙斯盾", 4),
    ("Mk41", 3), ("MK41", 3), ("带相控阵雷达", 5),
    ("航母的带刀护卫", 12), ("带刀护卫", 10),
    ("后面有直升机库", 8), ("两座直升机库", 16),
])

# 尼米兹级补充自然语言支持，避免“航母大类对但 class 为空”。
_v27_add_sig(KNOWN_CLASS_SUPPORT_SIGNATURES, "尼米兹级航空母舰", [
    ("特别大的军舰", 4), ("甲板特别宽", 7), ("多架固定翼飞机", 8),
    ("甲板上停放有多架固定翼飞机", 10), ("大型水面舰艇", 2),
    ("核动力的", 8), ("不用经常加油", 5), ("不用加油", 5),
    ("能搭载很多飞机", 8), ("有弹射器", 10),
])

# 提康德罗加级补充中等支持，但仍不使用未知类特征。
_v27_add_sig(KNOWN_CLASS_SUPPORT_SIGNATURES, "提康德罗加级导弹巡洋舰", [
    ("带宙斯盾的巡洋舰", 12), ("四个大平板雷达", 8),
    ("专门给航母护航", 8), ("前后都有垂发", 8),
    ("能打防空导弹和对地导弹", 5), ("水面作战舰艇", 2),
    ("舰体修长", 4), ("上层建筑较高", 4), ("比伯克级大一点", 10),
    ("能当指挥舰用", 12), ("AN/SPY-1", 8),
])

# 独立级补充“已知类自身”自然语言支持，不使用未知护卫舰特征。
_v27_add_sig(KNOWN_CLASS_SUPPORT_SIGNATURES, "独立级濒海战斗舰", [
    ("不是普通单体船", 10), ("左右支撑结构", 12), ("像三体船", 12),
    ("很科幻", 5), ("甲板很宽", 4), ("三个船身", 12),
])

# 两栖舰 / 登陆舰补充已知类支持，减少 open-set 误伤。
_v27_add_sig(KNOWN_CLASS_SUPPORT_SIGNATURES, "黄蜂级两栖攻击舰", [
    ("像小航母的军舰", 10), ("能停直升机和垂直起降的战斗机", 12),
    ("后面还有坞舱", 10), ("能当小航母用", 10),
    ("排水量四万多吨", 8), ("搭载垂直起降战斗机和气垫艇", 12),
])
_v27_add_sig(KNOWN_CLASS_SUPPORT_SIGNATURES, "圣安东尼奥级两栖船坞运输舰", [
    ("两栖运输舰", 12), ("美国的两栖运输舰", 14),
    ("有坞舱能放登陆艇", 10), ("也有直升机库", 8),
    ("上面有直升机甲板", 5), ("能运兵和车辆", 10),
    ("有坞舱和直升机库", 12), ("桅杆是一体化的", 8),
])
_v27_add_sig(KNOWN_CLASS_SUPPORT_SIGNATURES, "惠德比岛级船坞登陆舰", [
    ("船坞登陆舰", 12), ("大坞舱", 10), ("放好几艘气垫登陆艇", 10),
    ("上面有直升机平台但没有机库", 10), ("未观察到直升机库结构", 10),
    ("主要用来运输登陆装备", 10), ("用来放登陆艇", 9),
])

# 2) 重新生成强匹配表，确保上面的修改生效。
KNOWN_CLASS_STRONG_SIGNATURES.clear()
for _cls, _items in KNOWN_CLASS_ANCHOR_SIGNATURES.items():
    _v26_extend_sig(KNOWN_CLASS_STRONG_SIGNATURES, _cls, [(t, max(6.0, w * 0.65)) for t, w in _items])
for _cls, _items in KNOWN_CLASS_SUPPORT_SIGNATURES.items():
    _v26_extend_sig(KNOWN_CLASS_STRONG_SIGNATURES, _cls, [(t, max(3.0, w * 0.75)) for t, w in _items])

# 3) 通用大类提示：只用于大类判断，不用于匹配任何未知舰级。
CATEGORY_FEATURE_HINTS.setdefault("驱逐舰", {}).update({
    "航母的带刀护卫": 2.5, "带刀护卫": 2.0, "一艘驱逐舰": 2.5,
})
CATEGORY_FEATURE_HINTS.setdefault("护卫舰", {}).update({
    "护卫舰": 3.0, "巡防舰": 2.5, "反潜护卫舰": 3.0, "通用护卫舰": 2.5, "现代护卫舰": 2.0,
})
CATEGORY_FEATURE_HINTS.setdefault("航空母舰", {}).update({
    "固定翼飞机": 2.0, "多架固定翼飞机": 2.5, "有弹射器": 2.5,
})
