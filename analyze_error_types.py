# -*- coding: utf-8 -*-
"""
analyze_error_types.py

用途：
  读取 replay_*_all_errors.csv 与对应 sim_text_inputs*.jsonl，
  给每条错误样本补充 input_text、source_type、description_level、noise_level，
  并按错误路径/数据构造问题做启发式诊断。

典型用法（PowerShell）：
.venv\Scripts\python.exe .\analyze_error_types.py `
  --errors replay_v49_12_all_errors.csv `
  --inputs sim_text_inputs_v2_80.jsonl sim_text_inputs_test_100_fixed.jsonl sim_text_inputs_dev_anchor_80.jsonl `
  --output replay_v49_12_error_diagnosis.csv `
  --report replay_v49_12_error_diagnosis_report.md

说明：
  这个脚本不调用 LLM，不修改 run_deepseek.py，只做错误样本诊断。
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple


def norm_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"true", "1", "yes", "y", "是"}


def safe_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and str(v) == "nan":
        return ""
    return str(v).strip()


def read_inputs(paths: List[str]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"[WARN] input 文件不存在，跳过: {p}")
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                sid = safe_text(obj.get("id"))
                if not sid:
                    continue
                result[sid] = {
                    "input_text": safe_text(obj.get("input_text") or obj.get("text") or obj.get("prompt")),
                    "source_type": safe_text(obj.get("source_type")),
                    "description_level": safe_text(obj.get("description_level")),
                    "noise_level": safe_text(obj.get("noise_level")),
                    "input_file": path.name,
                }
    return result


def contains_any(text: str, terms: List[str]) -> bool:
    return any(t and t in text for t in terms)


def count_any(text: str, terms: List[str]) -> int:
    return sum(1 for t in terms if t and t in text)


def regex_any(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text, flags=re.I) for p in patterns)


ANCHORS = {
    "尼米兹级航空母舰": ["尼米兹", "CVN-68", "核动力航空母舰", "蒸汽弹射", "弹射器", "拦阻索", "10万吨", "100000吨", "全通飞行甲板", "斜角飞行甲板"],
    "提康德罗加级导弹巡洋舰": ["提康德罗加", "CG-47", "122单元", "122 单元", "Mk41", "MK41", "舰队指挥", "巡洋舰"],
    "阿利·伯克级驱逐舰": ["阿利·伯克", "阿利伯克", "DDG-51", "DDG51", "Flight IIA", "Flight III", "宙斯盾", "Aegis", "SPY-1", "SPY-6", "SPY", "Mk41", "MK41", "96单元", "96 单元", "90单元", "90 单元"],
    "独立级濒海战斗舰": ["独立级", "LCS-2", "LCS", "三体", "三体船", "57mm", "57毫米", "模块化任务", "任务模块", "近海作战", "濒海", "RAM", "MH-60"],
    "黄蜂级两栖攻击舰": ["黄蜂级", "LHD", "两栖攻击舰", "全通飞行甲板", "STOVL", "短距起飞", "垂直降落", "AV-8B", "F-35B", "航空突击"],
    "圣安东尼奥级两栖船坞运输舰": ["圣安东尼奥", "LPD-17", "LPD", "两栖船坞运输", "船坞运输舰", "车辆甲板", "货物", "人员", "综合运输"],
    "惠德比岛级船坞登陆舰": ["惠德比", "LSD-41", "LSD", "船坞登陆舰", "大型坞舱", "气垫登陆艇", "LCAC", "登陆艇投送", "艉门"],
}

SHARED_DISTINGUISHING = {
    "surface_combatant_shared": ["相控阵", "垂直发射", "垂发", "主炮", "导弹", "区域防空", "防空", "雷达阵面", "舰艏主炮"],
    "frigate_littoral_shared": ["三体", "57mm", "57毫米", "模块化", "近海", "濒海", "直升机甲板", "RAM", "MH-60"],
    "amphibious_landing_shared": ["坞舱", "艉门", "登陆艇", "LCAC", "气垫登陆艇", "车辆甲板", "运兵", "两栖", "登陆作战"],
    "aviation_shared": ["飞行甲板", "直升机", "机库", "升降机", "航空作业", "舰载机"],
}

WEAK_TERMS = ["隐身", "简洁", "上层建筑", "雷达", "桅杆", "舰桥", "舰体线条", "倾斜", "直升机平台", "大型舰艇", "现代化"]

NEGATIVE_MARKERS = ["无", "没有", "未", "不具备", "未观察到", "不明显", "缺少", "不是", "非"]


def has_negative_near(text: str, term: str, window: int = 6) -> bool:
    idx = text.find(term)
    if idx < 0:
        return False
    left = text[max(0, idx - window):idx]
    return any(m in left for m in NEGATIVE_MARKERS)


def numeric_profile(text: str) -> Dict[str, Any]:
    nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*(?:米|吨|节|海里|人|架|艘)", text)]
    has_param = bool(nums) or regex_any(text, [r"\d+\s*米", r"\d+\s*吨", r"\d+\s*节", r"\d+\s*海里", r"\d+\s*人"])
    # 粗略识别中型护卫舰/驱逐舰参数段：120-170m, 3000-8000吨
    medium_length = regex_any(text, [r"1[2-6]\d(?:\.\d+)?\s*米", r"15\d(?:\.\d+)?\s*米"])
    medium_disp = regex_any(text, [r"[3-7]\d{3}\s*吨", r"[3-7],\d{3}\s*吨"])
    huge_carrier = regex_any(text, [r"10\s*万\s*吨", r"100000\s*吨", r"104000\s*吨", r"332(?:\.8)?\s*米"])
    return {
        "has_parameter_text": has_param,
        "medium_ship_parameters": medium_length or medium_disp,
        "carrier_scale_parameters": huge_carrier,
    }


def evidence_flags(text: str, gold_known: str = "", pred_known: str = "") -> Dict[str, Any]:
    flags: Dict[str, Any] = {}
    flags.update(numeric_profile(text))

    for name, terms in SHARED_DISTINGUISHING.items():
        flags[name] = count_any(text, terms)

    flags["weak_feature_count"] = count_any(text, WEAK_TERMS)
    flags["negative_expression_count"] = count_any(text, NEGATIVE_MARKERS)

    if gold_known:
        flags["gold_anchor_count"] = count_any(text, ANCHORS.get(gold_known, []))
    else:
        flags["gold_anchor_count"] = 0
    if pred_known:
        flags["pred_anchor_count"] = count_any(text, ANCHORS.get(pred_known, []))
    else:
        flags["pred_anchor_count"] = 0

    # 反向排除示例：出现“无全通飞行甲板/无坞舱/无垂发”等
    neg_hits = []
    for term in ["全通飞行甲板", "飞行甲板", "坞舱", "艉门", "垂发", "垂直发射", "弹射器", "拦阻索"]:
        if term in text and has_negative_near(text, term):
            neg_hits.append(term)
    flags["negative_near_terms"] = "|".join(neg_hits)
    return flags


def error_type(row: Dict[str, Any]) -> str:
    gold_open = norm_bool(row.get("gold_open_set"))
    pred_open = norm_bool(row.get("pred_open_set"))
    gold_cat = safe_text(row.get("gold_category"))
    pred_cat = safe_text(row.get("pred_category"))
    gold_known = safe_text(row.get("gold_known_class"))
    pred_known = safe_text(row.get("pred_known_class"))

    if gold_open and not pred_open:
        return "unknown_to_known_closed_set_leak"
    if (not gold_open) and pred_open:
        if gold_cat == pred_cat:
            return "known_to_unknown_same_category_rejection"
        return "known_to_unknown_wrong_category_rejection"
    if gold_open and pred_open:
        if gold_cat != pred_cat:
            return "unknown_to_unknown_wrong_category"
        return "unknown_open_set_other_error"
    if (not gold_open) and (not pred_open):
        if gold_known != pred_known:
            if gold_cat != pred_cat:
                return "known_to_known_wrong_category_class"
            return "known_to_known_same_category_wrong_class"
    return "other"


def diagnose(row: Dict[str, Any], text: str, flags: Dict[str, Any]) -> Tuple[str, str]:
    et = error_type(row)
    gold_cat = safe_text(row.get("gold_category"))
    pred_cat = safe_text(row.get("pred_category"))
    gold_known = safe_text(row.get("gold_known_class"))
    pred_known = safe_text(row.get("pred_known_class"))

    notes = []
    main = et

    if et == "unknown_to_known_closed_set_leak":
        if flags.get("pred_anchor_count", 0) >= 2:
            main = "unknown_sample_contains_known_like_anchors"
            notes.append("未知类文本含多个预测已知类锚点，数据近邻过强或金标边界需复查。")
        elif flags.get("surface_combatant_shared", 0) >= 2 or flags.get("frigate_littoral_shared", 0) >= 2 or flags.get("amphibious_landing_shared", 0) >= 2:
            main = "shared_distinguishing_features_used_as_closed_set"
            notes.append("共享区分特征被用于闭集舰级确认，应只支持大类/方向。")
        else:
            main = "closed_set_leak_unclear_evidence"
            notes.append("未知类被闭集化，但文本锚点不明显，需看槽位或候选分。")

    elif et.startswith("known_to_unknown"):
        if flags.get("gold_anchor_count", 0) == 0:
            main = "known_sample_missing_unique_anchor"
            notes.append("真实已知类文本缺少独有强锚点，规则收紧时容易被拒识。")
        elif flags.get("gold_anchor_count", 0) == 1 and flags.get("weak_feature_count", 0) >= 2:
            main = "known_sample_anchor_too_weak_or_vlm_style"
            notes.append("有少量锚点但整体偏弱描述/VLM描述，容易触发类别内未知。")
        else:
            main = "rule_too_conservative_despite_anchors"
            notes.append("文本已有已知类锚点，但规则仍拒识，可能需在该舰级恢复条件中补组合证据。")

    elif et == "unknown_to_unknown_wrong_category":
        if flags.get("has_parameter_text") and flags.get("medium_ship_parameters"):
            main = "parameter_boundary_ambiguous"
            notes.append("参数型中型舰描述容易在护卫舰/驱逐舰/巡洋舰间漂移。")
        elif gold_cat in {"两栖舰", "登陆舰"} or pred_cat in {"两栖舰", "登陆舰"} or flags.get("amphibious_landing_shared", 0) >= 2:
            main = "amphibious_landing_category_boundary"
            notes.append("两栖/登陆共享特征导致大类边界混淆。")
        elif flags.get("surface_combatant_shared", 0) >= 2:
            main = "surface_combatant_category_boundary"
            notes.append("水面作战舰共享特征导致护卫舰/驱逐舰/巡洋舰边界混淆。")
        elif flags.get("weak_feature_count", 0) >= 2:
            main = "weak_description_insufficient_for_category"
            notes.append("文本只提供弱视觉特征，不足以稳定判定具体大类。")
        else:
            main = "unknown_category_boundary_unclear"
            notes.append("未知类大类错分，需结合槽位查看主导证据。")

    elif et.startswith("known_to_known"):
        if gold_cat in {"两栖舰", "登陆舰"} or pred_cat in {"两栖舰", "登陆舰"}:
            main = "amphibious_landing_known_class_confusion"
            notes.append("黄蜂/圣安东尼奥/惠德比岛边界混淆。")
        elif flags.get("gold_anchor_count", 0) == 0:
            main = "known_class_confusion_due_to_missing_anchor"
            notes.append("真实已知类锚点缺失，导致与近邻已知类混淆。")
        else:
            main = "known_class_confusion_near_neighbor"
            notes.append("已知类近邻混淆，需要更强反证或组合证据。")

    # 数据设计提示
    design_notes = []
    if safe_text(row.get("gold_open_set")).lower() == "true" and flags.get("pred_anchor_count", 0) >= 2:
        design_notes.append("未知样本含预测已知类强锚点，可能过像已知类。")
    if safe_text(row.get("gold_open_set")).lower() == "false" and flags.get("gold_anchor_count", 0) == 0:
        design_notes.append("已知样本缺少该舰级独有强锚点。")
    if flags.get("weak_feature_count", 0) >= 3 and not flags.get("has_parameter_text"):
        design_notes.append("弱视觉描述较多，判别信息不足。")
    if design_notes:
        notes.append("数据设计提示：" + "；".join(design_notes))

    return main, "；".join(notes)


def read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_report(path: str, rows: List[Dict[str, Any]]) -> None:
    total = len(rows)
    by_error = Counter(r.get("error_type", "") for r in rows)
    by_diag = Counter(r.get("diagnosis", "") for r in rows)
    by_dataset = Counter(r.get("dataset", "") for r in rows)
    by_source = Counter(r.get("source_type", "") for r in rows)
    by_gold = Counter(r.get("gold_category", "") for r in rows)

    lines = []
    lines.append(f"# Error Diagnosis Report\n")
    lines.append(f"Total error rows: **{total}**\n")

    def section(title: str, counter: Counter):
        lines.append(f"\n## {title}\n")
        lines.append("| item | count |\n|---|---:|\n")
        for k, v in counter.most_common():
            lines.append(f"| {k or '(empty)'} | {v} |\n")

    section("By original error type", by_error)
    section("By diagnosis", by_diag)
    section("By dataset", by_dataset)
    section("By source_type", by_source)
    section("By gold_category", by_gold)

    lines.append("\n## Suggested next action\n")
    lines.append("- 如果 `known_sample_missing_unique_anchor` 较多：优先检查已知类样本文本是否缺少独有强锚点，而不是继续放宽规则。\n")
    lines.append("- 如果 `shared_distinguishing_features_used_as_closed_set` 较多：说明共享区分特征仍被用于闭集舰级确认，应继续限制 known_class 恢复路径。\n")
    lines.append("- 如果 `parameter_boundary_ambiguous` 较多：需要在规则中增加长度/排水量等参数型边界。\n")
    lines.append("- 如果 `amphibious_landing_*` 较多：继续微调黄蜂/圣安东尼奥/惠德比岛边界。\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors", required=True, help="replay_*_all_errors.csv")
    ap.add_argument("--inputs", nargs="+", required=True, help="sim_text_inputs*.jsonl，可传多个")
    ap.add_argument("--output", default="error_diagnosis.csv")
    ap.add_argument("--report", default="error_diagnosis_report.md")
    args = ap.parse_args()

    input_map = read_inputs(args.inputs)
    rows = read_csv(args.errors)
    out_rows = []

    for r in rows:
        sid = safe_text(r.get("id"))
        meta = input_map.get(sid, {})
        text = meta.get("input_text", "")
        gold_known = safe_text(r.get("gold_known_class"))
        pred_known = safe_text(r.get("pred_known_class"))
        flags = evidence_flags(text, gold_known, pred_known)
        et = error_type(r)
        diag, note = diagnose(r, text, flags)

        new = dict(r)
        new.update({
            "error_type": et,
            "diagnosis": diag,
            "diagnosis_note": note,
            "input_text": text,
            "source_type": meta.get("source_type", ""),
            "description_level": meta.get("description_level", ""),
            "noise_level": meta.get("noise_level", ""),
            "input_file": meta.get("input_file", ""),
        })
        for k, v in flags.items():
            new[k] = v
        out_rows.append(new)

    write_csv(args.output, out_rows)
    write_report(args.report, out_rows)

    print(f"[OK] wrote diagnosis csv: {args.output}")
    print(f"[OK] wrote report: {args.report}")
    print(f"[INFO] errors: {len(out_rows)}, matched input_text: {sum(1 for r in out_rows if r.get('input_text'))}")


if __name__ == "__main__":
    main()
