# -*- coding: utf-8 -*-
"""
diagnose_errors.py

用途：
把 replay_summary.csv 中的错误样本，与 predictions_detail.jsonl 中的原始输入、
LLM observed_attributes、match_result、prediction 合并，生成便于人工诊断的 CSV。

用法：
python diagnose_errors.py <predictions_detail.jsonl> <replay_summary.csv> <output.csv>
"""

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    data: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = row.get("id")
            if sample_id:
                data[sample_id] = row
    return data


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def get_nested(d: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


def value_to_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return "; ".join(value_to_text(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def slot(obs: Dict[str, Any], group: str, name: str) -> str:
    return value_to_text(get_nested(obs, group, name, default=""))


def first_existing(row: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            return value_to_text(v)
    return ""


def short_json(obj: Any, max_len: int = 1200) -> str:
    text = value_to_text(obj)
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def main() -> None:
    if len(sys.argv) != 4:
        print("用法: python diagnose_errors.py <predictions_detail.jsonl> <replay_summary.csv> <output.csv>")
        sys.exit(1)

    detail_path = Path(sys.argv[1])
    summary_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])

    if not detail_path.exists():
        raise FileNotFoundError(f"detail 文件不存在: {detail_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"summary 文件不存在: {summary_path}")

    details = read_jsonl(detail_path)
    summary_rows = read_csv(summary_path)

    wrong_rows = [r for r in summary_rows if str(r.get("exact_correct", "")).lower() != "true"]

    fields = [
        "id",
        "source_type",
        "description_level",
        "noise_level",
        "gold_category",
        "gold_known_class",
        "gold_open_set",
        "pred_category",
        "pred_known_class",
        "pred_open_set",
        "exact_correct",
        "input_text",

        # 航空/两栖关键卡槽
        "Flight_Deck_Type",
        "Flight_Deck_Position",
        "Catapult",
        "Catapult_Count",
        "Arresting_Gear",
        "Arresting_Gear_Count",
        "Aircraft_Capacity_Level",
        "Fixed_Wing_Aircraft_Operation",
        "STOVL_Aircraft_Operation",
        "Hangar",
        "Well_Deck",
        "Stern_Gate",
        "Landing_Craft_Capability",
        "Landing_Craft_Capacity",
        "Vehicle_Deck",
        "Troop_Transport",

        # 结构/武器/任务关键卡槽
        "Hull_Form",
        "Superstructure_Type",
        "Island_Position",
        "Mast_Feature",
        "Stealth_Shape",
        "VLS_Presence",
        "VLS_Count_Level",
        "VLS_Position",
        "Main_Gun_Presence",
        "Main_Gun_Position",
        "Main_Gun_Caliber",
        "Phased_Array_Radar",
        "Radar_Array_Type",
        "Anti_Ship_Missile_Launcher",
        "Sonar_Feature",
        "Primary_Mission",
        "Area_Air_Defense",
        "Anti_Submarine",
        "Amphibious_Assault",
        "Landing_Operation",
        "Command_Control",
        "Fleet_Core",
        "Patrol_Littoral",

        # 参数/装备关键卡槽
        "Length_Overall",
        "Beam",
        "Full_Load_Displacement",
        "Speed",
        "Range",
        "Crew",
        "Aircraft_Capacity",
        "Troop_Capacity",
        "Power_Output",
        "Propulsion",
        "Powerplant",
        "Radar_System",
        "Combat_System",
        "Weapon_System",
        "Aircraft",
        "Landing_Craft",
        "Mission_Module",
        "Keywords",

        # 便于定位后处理问题
        "prediction_json",
        "match_result_json",
        "v29_slot_based_analysis",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for r in wrong_rows:
            sample_id = r.get("id", "")
            d = details.get(sample_id, {})
            obs = d.get("observed_attributes", {}) if isinstance(d, dict) else {}

            match_result = d.get("match_result", {})
            prediction = d.get("prediction", {})
            analysis = ""
            if isinstance(match_result, dict):
                analysis = value_to_text(match_result.get("v29_slot_based_analysis", ""))
            if not analysis and isinstance(prediction, dict):
                analysis = value_to_text(prediction.get("v29_slot_based_analysis", ""))

            out = {
                "id": sample_id,
                "source_type": d.get("source_type", r.get("source_type", "")),
                "description_level": d.get("description_level", r.get("description_level", "")),
                "noise_level": d.get("noise_level", r.get("noise_level", "")),
                "gold_category": r.get("gold_category", get_nested(d, "gold", "gold_category", default="")),
                "gold_known_class": r.get("gold_known_class", get_nested(d, "gold", "gold_known_class", default="")),
                "gold_open_set": r.get("gold_open_set", get_nested(d, "gold", "gold_open_set", default="")),
                "pred_category": r.get("pred_category", first_existing(prediction, ["category", "pred_category"])),
                "pred_known_class": r.get("pred_known_class", first_existing(prediction, ["known_class", "pred_known_class"])),
                "pred_open_set": r.get("pred_open_set", first_existing(prediction, ["open_set", "pred_open_set"])),
                "exact_correct": r.get("exact_correct", ""),
                "input_text": d.get("input_text", ""),

                "Flight_Deck_Type": slot(obs, "AVIATION_FEATURES", "Flight_Deck_Type"),
                "Flight_Deck_Position": slot(obs, "AVIATION_FEATURES", "Flight_Deck_Position"),
                "Catapult": slot(obs, "AVIATION_FEATURES", "Catapult"),
                "Catapult_Count": slot(obs, "AVIATION_FEATURES", "Catapult_Count"),
                "Arresting_Gear": slot(obs, "AVIATION_FEATURES", "Arresting_Gear"),
                "Arresting_Gear_Count": slot(obs, "AVIATION_FEATURES", "Arresting_Gear_Count"),
                "Aircraft_Capacity_Level": slot(obs, "AVIATION_FEATURES", "Aircraft_Capacity_Level"),
                "Fixed_Wing_Aircraft_Operation": slot(obs, "AVIATION_FEATURES", "Fixed_Wing_Aircraft_Operation"),
                "STOVL_Aircraft_Operation": slot(obs, "AVIATION_FEATURES", "STOVL_Aircraft_Operation"),
                "Hangar": slot(obs, "AVIATION_FEATURES", "Hangar"),
                "Well_Deck": slot(obs, "AMPHIBIOUS_FEATURES", "Well_Deck"),
                "Stern_Gate": slot(obs, "AMPHIBIOUS_FEATURES", "Stern_Gate"),
                "Landing_Craft_Capability": slot(obs, "AMPHIBIOUS_FEATURES", "Landing_Craft_Capability"),
                "Landing_Craft_Capacity": slot(obs, "AMPHIBIOUS_FEATURES", "Landing_Craft_Capacity"),
                "Vehicle_Deck": slot(obs, "AMPHIBIOUS_FEATURES", "Vehicle_Deck"),
                "Troop_Transport": slot(obs, "AMPHIBIOUS_FEATURES", "Troop_Transport"),

                "Hull_Form": slot(obs, "VISUAL_STRUCTURE", "Hull_Form"),
                "Superstructure_Type": slot(obs, "VISUAL_STRUCTURE", "Superstructure_Type"),
                "Island_Position": slot(obs, "VISUAL_STRUCTURE", "Island_Position"),
                "Mast_Feature": slot(obs, "VISUAL_STRUCTURE", "Mast_Feature"),
                "Stealth_Shape": slot(obs, "VISUAL_STRUCTURE", "Stealth_Shape"),
                "VLS_Presence": slot(obs, "WEAPON_SENSOR_FEATURES", "VLS_Presence"),
                "VLS_Count_Level": slot(obs, "WEAPON_SENSOR_FEATURES", "VLS_Count_Level"),
                "VLS_Position": slot(obs, "WEAPON_SENSOR_FEATURES", "VLS_Position"),
                "Main_Gun_Presence": slot(obs, "WEAPON_SENSOR_FEATURES", "Main_Gun_Presence"),
                "Main_Gun_Position": slot(obs, "WEAPON_SENSOR_FEATURES", "Main_Gun_Position"),
                "Main_Gun_Caliber": slot(obs, "WEAPON_SENSOR_FEATURES", "Main_Gun_Caliber"),
                "Phased_Array_Radar": slot(obs, "WEAPON_SENSOR_FEATURES", "Phased_Array_Radar"),
                "Radar_Array_Type": slot(obs, "WEAPON_SENSOR_FEATURES", "Radar_Array_Type"),
                "Anti_Ship_Missile_Launcher": slot(obs, "WEAPON_SENSOR_FEATURES", "Anti_Ship_Missile_Launcher"),
                "Sonar_Feature": slot(obs, "WEAPON_SENSOR_FEATURES", "Sonar_Feature"),
                "Primary_Mission": slot(obs, "MISSION_FEATURES", "Primary_Mission"),
                "Area_Air_Defense": slot(obs, "MISSION_FEATURES", "Area_Air_Defense"),
                "Anti_Submarine": slot(obs, "MISSION_FEATURES", "Anti_Submarine"),
                "Amphibious_Assault": slot(obs, "MISSION_FEATURES", "Amphibious_Assault"),
                "Landing_Operation": slot(obs, "MISSION_FEATURES", "Landing_Operation"),
                "Command_Control": slot(obs, "MISSION_FEATURES", "Command_Control"),
                "Fleet_Core": slot(obs, "MISSION_FEATURES", "Fleet_Core"),
                "Patrol_Littoral": slot(obs, "MISSION_FEATURES", "Patrol_Littoral"),

                "Length_Overall": slot(obs, "TEXT_ATTRIBUTES", "Length_Overall"),
                "Beam": slot(obs, "TEXT_ATTRIBUTES", "Beam"),
                "Full_Load_Displacement": slot(obs, "TEXT_ATTRIBUTES", "Full_Load_Displacement"),
                "Speed": slot(obs, "TEXT_ATTRIBUTES", "Speed"),
                "Range": slot(obs, "TEXT_ATTRIBUTES", "Range"),
                "Crew": slot(obs, "TEXT_ATTRIBUTES", "Crew"),
                "Aircraft_Capacity": slot(obs, "TEXT_ATTRIBUTES", "Aircraft_Capacity"),
                "Troop_Capacity": slot(obs, "TEXT_ATTRIBUTES", "Troop_Capacity"),
                "Power_Output": slot(obs, "TEXT_ATTRIBUTES", "Power_Output"),
                "Propulsion": slot(obs, "TEXT_ATTRIBUTES", "Propulsion"),
                "Powerplant": slot(obs, "TEXT_ATTRIBUTES", "Powerplant"),
                "Radar_System": slot(obs, "EQUIPMENT_DETAILS", "Radar_System"),
                "Combat_System": slot(obs, "EQUIPMENT_DETAILS", "Combat_System"),
                "Weapon_System": slot(obs, "EQUIPMENT_DETAILS", "Weapon_System"),
                "Aircraft": slot(obs, "EQUIPMENT_DETAILS", "Aircraft"),
                "Landing_Craft": slot(obs, "EQUIPMENT_DETAILS", "Landing_Craft"),
                "Mission_Module": slot(obs, "EQUIPMENT_DETAILS", "Mission_Module"),
                "Keywords": slot(obs, "TEXT_STRONG_CUES", "Keywords"),

                "prediction_json": short_json(prediction),
                "match_result_json": short_json(match_result),
                "v29_slot_based_analysis": short_json(analysis, max_len=2000),
            }
            writer.writerow(out)

    print(f"错误样本数: {len(wrong_rows)}")
    print(f"输出完成: {output_path}")


if __name__ == "__main__":
    main()
