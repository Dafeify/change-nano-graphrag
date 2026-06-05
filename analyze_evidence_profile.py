# -*- coding: utf-8 -*-
"""
analyze_evidence_profile.py

用途：
  对 replay_*_all_errors.csv 中的错误样本做 evidence profile 诊断。
  重点区分：
    1) 显式名称证据 explicit_name_evidence
    2) 独有强锚点 unique_anchor
    3) 共享区分特征 shared_distinguishing
    4) 普通弱特征 weak_feature
    5) 反向排除/冲突证据 negative_conflict
    6) 组合证据 combo_evidence

运行示例：
.venv\Scripts\python.exe .\analyze_evidence_profile.py `
  --errors .\replay_v49_16_all_errors.csv `
  --inputs .\sim_text_inputs_v2_80.jsonl .\sim_text_inputs_test_100_fixed.jsonl .\sim_text_inputs_dev_anchor_80.jsonl `
  --details .\batch_results_v17_v2_80\predictions_detail.jsonl .\batch_results_test_100\predictions_detail.jsonl .\batch_results_dev_anchor_80\predictions_detail.jsonl `
  --output .\replay_v49_16_evidence_profile.csv `
  --report .\replay_v49_16_evidence_profile_report.md
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

UNKNOWN_LIKE = {"", "未知", "None", "null", "类别内未知类"}

# 每个已知类的证据定义。注意：显式名称不是一票决定，只是高权重证据，仍会被冲突证据降级。
CLASS_RULES: Dict[str, Dict[str, List[str]]] = {
    "尼米兹级航空母舰": {
        "explicit": ["尼米兹", "nimitz", "cvn-68", "cvn68", "cvn-69", "cvn-70", "cvn-71", "cvn-72", "cvn-73", "cvn-74", "cvn-75", "cvn-76", "cvn-77"],
        "unique": ["蒸汽弹射", "弹射器", "拦阻索", "核动力航空母舰", "十万吨", "10万吨", "100000吨", "超过10万", "cvn"],
        "shared": ["全通飞行甲板", "斜角飞行甲板", "舰岛", "舰载机", "升降机", "航母战斗群"],
        "weak": ["航空作业", "远洋", "大型舰艇", "舰队核心"],
        "negative": ["无弹射器", "没有弹射器", "无拦阻", "没有拦阻", "无全通飞行甲板", "没有全通飞行甲板", "短距起飞", "垂直降落", "stovl", "坞舱", "登陆艇"],
    },
    "提康德罗加级导弹巡洋舰": {
        "explicit": ["提康德罗加", "ticonderoga", "cg-47", "cg47"],
        "unique": ["122单元", "一百二十二单元", "双臂发射器", "舰队指挥", "巡洋舰旗舰", "cg-"],
        "shared": ["mk41", "mk-41", "垂直发射", "垂发", "宙斯盾", "aegis", "spy-1", "相控阵", "区域防空"],
        "weak": ["防空", "反潜", "巡洋舰", "舰队", "导弹舰"],
        "negative": ["三体", "57mm", "近海", "模块化", "坞舱", "登陆艇", "全通飞行甲板", "弹射器", "拦阻索", "排水量3650", "3650吨", "4550吨", "6200吨"],
    },
    "阿利·伯克级驱逐舰": {
        "explicit": ["阿利·伯克", "阿利伯克", "arleigh burke", "ddg-51", "ddg51", "flight iia", "flight iii", "flight ii", "伯克级"],
        "unique": ["spy-1", "spy-6", "宙斯盾", "aegis", "mk41", "mk-41", "90单元", "九十单元", "96单元", "九十六单元", "前后垂发", "直升机机库", "双机库"],
        "shared": ["相控阵", "垂直发射", "垂发", "舰艏主炮", "127mm", "127毫米", "区域防空", "导弹驱逐舰"],
        "weak": ["防空", "反潜", "多用途", "隐身化", "现代化驱逐舰", "直升机甲板"],
        "negative": ["三体", "57mm", "57毫米", "模块化", "近海", "lcs", "坞舱", "登陆艇", "全通飞行甲板", "弹射器", "拦阻索", "少量垂发", "有限垂发", "无垂发", "没有垂发"],
    },
    "独立级濒海战斗舰": {
        "explicit": ["独立级", "independence", "lcs-2", "lcs2"],
        "unique": ["三体", "三体船", "三体结构", "宽体三体", "两侧支撑船体", "多体船"],
        "shared": ["57mm", "57毫米", "模块化", "任务模块", "近海作战", "濒海作战", "ram", "拉姆", "mh-60", "直升机平台", "宽大舰尾", "高速近海"],
        "weak": ["护卫舰", "巡逻", "反潜", "反水雷", "小型舰艇", "高速"],
        "negative": ["127mm", "127毫米", "90单元", "96单元", "122单元", "宙斯盾", "spy-1", "spy-6", "mk41", "大型垂发", "区域防空", "坞舱", "登陆艇", "全通飞行甲板"],
    },
    "黄蜂级两栖攻击舰": {
        "explicit": ["黄蜂级", "wasp", "lhd-1", "lhd1"],
        "unique": ["lhd", "两栖攻击舰", "全通飞行甲板", "贯通飞行甲板", "stovl", "短距起飞", "垂直降落", "av-8b", "f-35b", "航空突击"],
        "shared": ["坞舱", "登陆艇", "直升机", "机库", "车辆甲板", "运兵", "两栖作战"],
        "weak": ["两栖", "投送", "登陆", "大型甲板"],
        "negative": ["无全通飞行甲板", "没有全通飞行甲板", "无大型航空设施", "船坞登陆舰", "lsd", "登陆艇投送为主", "多艘气垫登陆艇"],
    },
    "圣安东尼奥级两栖船坞运输舰": {
        "explicit": ["圣安东尼奥", "san antonio", "lpd-17", "lpd17"],
        "unique": ["lpd", "两栖船坞运输舰", "船坞运输", "lp d", "综合运输"],
        "shared": ["坞舱", "艉门", "登陆艇", "车辆甲板", "运兵", "直升机平台", "机库", "lcac"],
        "weak": ["两栖", "运输", "投送", "登陆"],
        "negative": ["全通飞行甲板", "stovl", "短距起飞", "垂直降落", "av-8b", "f-35b", "lsd", "船坞登陆舰", "登陆艇投送为主"],
    },
    "惠德比岛级船坞登陆舰": {
        "explicit": ["惠德比", "whidbey", "lsd-41", "lsd41"],
        "unique": ["lsd", "船坞登陆舰", "登陆艇投送为主", "大型坞舱", "多艘气垫登陆艇", "4艘lcac", "四艘lcac", "登陆艇收放"],
        "shared": ["坞舱", "艉门", "登陆艇", "lcac", "气垫登陆艇", "车辆甲板", "运兵"],
        "weak": ["两栖", "登陆", "运输", "投送"],
        "negative": ["全通飞行甲板", "stovl", "短距起飞", "垂直降落", "av-8b", "f-35b", "lpd", "两栖船坞运输舰", "前部大型上层建筑"],
    },
}

CATEGORY_WORDS = {
    "航空母舰": ["航空母舰", "航母", "cvn", "舰载机", "弹射器", "拦阻索", "全通飞行甲板"],
    "巡洋舰": ["巡洋舰", "舰队指挥", "122单元", "一百二十二单元", "万吨级", "区域防空旗舰"],
    "驱逐舰": ["驱逐舰", "导弹驱逐舰", "防空驱逐舰", "大型防空", "ddg"],
    "护卫舰": ["护卫舰", "濒海战斗舰", "近海作战", "lcs", "三体", "57mm"],
    "两栖舰": ["两栖攻击舰", "lhd", "stovl", "短距起飞", "垂直降落", "全通飞行甲板"],
    "登陆舰": ["船坞登陆舰", "lsd", "登陆艇投送", "大型坞舱", "气垫登陆艇"],
}


def norm_text(s: Any) -> str:
    text = str(s or "").lower()
    text = text.replace("－", "-").replace("—", "-").replace("–", "-")
    text = re.sub(r"\s+", "", text)
    return text


def contains_any(text: str, terms: List[str]) -> List[str]:
    hits = []
    for t in terms:
        nt = norm_text(t)
        if nt and nt in text:
            hits.append(t)
    return hits


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_inputs(paths: List[str]) -> Dict[str, str]:
    mp = {}
    for p in paths:
        path = Path(p)
        for row in read_jsonl(path):
            sid = str(row.get("id") or row.get("sid") or "")
            txt = row.get("input_text") or row.get("text") or row.get("user_text") or row.get("content") or ""
            if sid and txt:
                mp[sid] = str(txt)
    return mp


def flatten_values(x: Any) -> List[str]:
    vals = []
    if isinstance(x, dict):
        for v in x.values():
            vals.extend(flatten_values(v))
    elif isinstance(x, list):
        for v in x:
            vals.extend(flatten_values(v))
    elif x is not None:
        vals.append(str(x))
    return vals


def load_details(paths: List[str]) -> Dict[str, Dict[str, Any]]:
    mp = {}
    for p in paths:
        path = Path(p)
        for row in read_jsonl(path):
            sid = str(row.get("id") or "")
            if sid:
                mp[sid] = row
    return mp


def infer_text_for_row(row: Dict[str, Any], input_map: Dict[str, str], detail_map: Dict[str, Dict[str, Any]]) -> str:
    sid = str(row.get("id") or "")
    parts = []
    if input_map.get(sid):
        parts.append(input_map[sid])
    d = detail_map.get(sid) or {}
    if d.get("input_text"):
        parts.append(str(d.get("input_text")))
    obs = d.get("observed_attributes")
    if obs:
        parts.extend(flatten_values(obs))
    if d.get("textual_summary"):
        parts.append(str(d.get("textual_summary")))
    # 去重保持顺序
    seen = set()
    out = []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return "\n".join(out)


def extract_numbers(text_raw: str) -> Dict[str, Any]:
    text = text_raw.replace(",", "")
    nums: Dict[str, Any] = {}
    # 长度
    m = re.search(r"(?:舰长|船长|全长|长度)\s*[约为约]?\s*(\d+(?:\.\d+)?)\s*(?:米|m)", text, flags=re.I)
    if m:
        nums["length_m"] = float(m.group(1))
    # 排水量，优先满载
    m = re.search(r"满载排水量\s*[约为约]?\s*(\d+(?:\.\d+)?)\s*(?:吨|t)", text, flags=re.I)
    if not m:
        m = re.search(r"排水量\s*[约为约]?\s*(\d+(?:\.\d+)?)\s*(?:吨|t)", text, flags=re.I)
    if m:
        nums["displacement_t"] = float(m.group(1))
    # 垂发单元
    m = re.search(r"(\d+)\s*(?:单元|具|组)?\s*(?:mk[- ]?41|垂发|垂直发射)", text, flags=re.I)
    if not m:
        m = re.search(r"(?:mk[- ]?41|垂发|垂直发射)[^\d]{0,8}(\d+)\s*(?:单元|具|组)", text, flags=re.I)
    if m:
        nums["vls_cells"] = int(m.group(1))
    return nums


def parameter_hint(nums: Dict[str, Any]) -> str:
    length = nums.get("length_m")
    disp = nums.get("displacement_t")
    vls = nums.get("vls_cells")
    hints = []
    if disp is not None:
        if disp <= 4500:
            hints.append("small_surface_frigate_scale")
        elif disp <= 7000:
            hints.append("medium_frigate_scale")
        elif disp <= 11000:
            hints.append("destroyer_scale")
        elif disp >= 12000:
            hints.append("cruiser_or_larger_scale")
    if length is not None:
        if length <= 135:
            hints.append("short_hull_frigate_scale")
        elif length <= 155:
            hints.append("mid_hull_frigate_or_destroyer_scale")
        elif length <= 180:
            hints.append("destroyer_hull_scale")
        elif length >= 250:
            hints.append("carrier_or_amphibious_large_deck_scale")
    if vls is not None:
        if vls <= 24:
            hints.append("low_vls_frigate_like")
        elif 80 <= vls <= 100:
            hints.append("destroyer_vls_scale")
        elif vls >= 120:
            hints.append("cruiser_vls_scale")
    return ";".join(hints)


def profile_for_class(text_raw: str, cls: str) -> Dict[str, Any]:
    text = norm_text(text_raw)
    rules = CLASS_RULES.get(cls)
    if not rules:
        return {
            "class": cls,
            "explicit_hits": "",
            "unique_hits": "",
            "shared_hits": "",
            "weak_hits": "",
            "negative_hits": "",
            "combo_hits": "",
            "evidence_level": "no_rule",
            "score": 0,
        }

    explicit = contains_any(text, rules.get("explicit", []))
    unique = contains_any(text, rules.get("unique", []))
    shared = contains_any(text, rules.get("shared", []))
    weak = contains_any(text, rules.get("weak", []))
    negative = contains_any(text, rules.get("negative", []))

    combos = []
    if cls == "阿利·伯克级驱逐舰":
        if ("mk41" in text or "mk-41" in text or "前后垂发" in text) and ("127mm" in text or "127毫米" in text or "舰艏主炮" in text) and ("机库" in text or "直升机" in text):
            combos.append("Mk41/前后垂发 + 127mm/舰艏主炮 + 直升机机库")
        if ("宙斯盾" in text or "aegis" in text or "spy" in text) and ("mk41" in text or "mk-41" in text or "90单元" in text or "96单元" in text):
            combos.append("宙斯盾/SPY + Mk41/90-96单元")
    elif cls == "独立级濒海战斗舰":
        aux_count = 0
        for group in [["57mm", "57毫米"], ["模块化", "任务模块", "近海", "濒海", "lcs"], ["ram", "拉姆", "mh-60", "直升机平台", "宽大舰尾"]]:
            if contains_any(text, group):
                aux_count += 1
        if contains_any(text, ["三体", "三体船", "三体结构", "两侧支撑船体", "多体船"]) and aux_count >= 2:
            combos.append(f"三体结构 + {aux_count}类独立级辅助证据")
        if contains_any(text, ["75人", "约75人", "4300海里", "4300海里/20节"]) and contains_any(text, ["57mm", "57毫米"]) and contains_any(text, ["ram", "拉姆", "mh-60"]):
            combos.append("75人/4300海里 + 57mm + RAM/MH-60")
    elif cls == "黄蜂级两栖攻击舰":
        if contains_any(text, ["全通飞行甲板", "贯通飞行甲板"]) and contains_any(text, ["stovl", "短距起飞", "垂直降落", "av-8b", "f-35b", "两栖攻击"]):
            combos.append("全通飞行甲板 + STOVL/两栖攻击")
    elif cls == "圣安东尼奥级两栖船坞运输舰":
        if contains_any(text, ["lpd", "船坞运输", "两栖船坞运输舰"]) and contains_any(text, ["车辆", "人员", "货物", "登陆艇", "坞舱"]):
            combos.append("LPD/船坞运输 + 综合投送")
    elif cls == "惠德比岛级船坞登陆舰":
        if contains_any(text, ["lsd", "船坞登陆舰", "登陆艇投送为主"]) and contains_any(text, ["大型坞舱", "lcac", "气垫登陆艇", "艉门"]):
            combos.append("LSD/船坞登陆 + 大型坞舱/LCAC")

    score = len(explicit) * 5 + len(unique) * 3 + len(shared) * 1.5 + len(weak) * 0.5 + len(combos) * 4 - len(negative) * 4

    if negative and not (explicit or combos):
        level = "conflicting_or_negative"
    elif explicit and not negative:
        level = "explicit_name_evidence"
    elif combos and not negative:
        level = "strong_combo_evidence"
    elif unique and len(shared) >= 1 and not negative:
        level = "unique_anchor_plus_shared"
    elif unique and not negative:
        level = "unique_anchor_only"
    elif len(shared) >= 2 and not unique:
        level = "shared_distinguishing_only"
    elif shared or weak:
        level = "weak_or_partial_evidence"
    else:
        level = "no_positive_evidence"

    return {
        "class": cls,
        "explicit_hits": "|".join(explicit),
        "unique_hits": "|".join(unique),
        "shared_hits": "|".join(shared),
        "weak_hits": "|".join(weak),
        "negative_hits": "|".join(negative),
        "combo_hits": "|".join(combos),
        "evidence_level": level,
        "score": round(score, 2),
    }


def category_profile(text_raw: str) -> str:
    text = norm_text(text_raw)
    parts = []
    for cat, terms in CATEGORY_WORDS.items():
        hits = contains_any(text, terms)
        if hits:
            parts.append(f"{cat}:{'|'.join(hits)}")
    return "; ".join(parts)


def classify_error_type(row: Dict[str, Any]) -> str:
    gold_open = str(row.get("gold_open_set", "")).lower() == "true"
    pred_open = str(row.get("pred_open_set", "")).lower() == "true"
    gold_cat = row.get("gold_category")
    pred_cat = row.get("pred_category")
    gold_cls = row.get("gold_known_class") or ""
    pred_cls = row.get("pred_known_class") or ""
    if gold_open and not pred_open and pred_cls:
        return "unknown_to_known_false_positive"
    if gold_open and pred_open and gold_cat != pred_cat:
        return "unknown_to_unknown_wrong_category"
    if not gold_open and pred_open:
        if gold_cat == pred_cat:
            return "known_to_unknown_same_category"
        return "known_to_unknown_wrong_category"
    if not gold_open and not pred_open and gold_cls != pred_cls:
        if gold_cat == pred_cat:
            return "known_to_known_wrong_class_same_category"
        return "known_to_known_wrong_category"
    return "other_error"


def suggest(row: Dict[str, Any], gold_prof: Dict[str, Any], pred_prof: Dict[str, Any], param_hint_text: str) -> str:
    et = row.get("diagnosis_error_type") or classify_error_type(row)
    gold_level = gold_prof.get("evidence_level", "")
    pred_level = pred_prof.get("evidence_level", "")
    if et == "known_to_unknown_same_category":
        if gold_level in {"explicit_name_evidence", "strong_combo_evidence", "unique_anchor_plus_shared"}:
            return "可尝试安全恢复：gold已知类有显式/组合/锚点证据；检查是否被过严open-set保护挡住"
        return "不建议强行恢复：gold已知类证据偏弱或仅共享特征，可能是数据弱描述"
    if et == "unknown_to_known_false_positive":
        if pred_level in {"shared_distinguishing_only", "weak_or_partial_evidence", "unique_anchor_only"}:
            return "应加强开放集保护：pred已知类证据不足，可能把共享区分特征当独有锚点"
        return "检查是否为未知近邻样本过像已知类；若有显式名称需看是否有反证"
    if et == "unknown_to_unknown_wrong_category":
        if param_hint_text:
            return f"优先做大类边界纠偏：参数提示={param_hint_text}"
        return "检查显式大类词与共享区分特征，适合category-only修正"
    if et.startswith("known_to_known"):
        return "已知类间混淆：检查同大类独有锚点、组合证据和反向排除特征"
    return "人工复查"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors", required=True, help="replay_*_all_errors.csv")
    ap.add_argument("--inputs", nargs="*", default=[], help="sim_text_inputs*.jsonl，可传多个")
    ap.add_argument("--details", nargs="*", default=[], help="predictions_detail.jsonl，可传多个")
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    input_map = load_inputs(args.inputs)
    detail_map = load_details(args.details)

    rows = []
    with Path(args.errors).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = str(row.get("id") or "")
            text_raw = infer_text_for_row(row, input_map, detail_map)
            nums = extract_numbers(text_raw)
            param_hint_text = parameter_hint(nums)
            et = classify_error_type(row)

            gold_cls = row.get("gold_known_class") or ""
            pred_cls = row.get("pred_known_class") or ""
            gold_prof = profile_for_class(text_raw, gold_cls) if gold_cls not in UNKNOWN_LIKE else {"evidence_level": "gold_open_set_or_no_known_class", "score": 0}
            pred_prof = profile_for_class(text_raw, pred_cls) if pred_cls not in UNKNOWN_LIKE else {"evidence_level": "pred_open_set_or_no_known_class", "score": 0}

            out = dict(row)
            out.update({
                "diagnosis_error_type": et,
                "input_text": input_map.get(sid, "") or (detail_map.get(sid, {}) or {}).get("input_text", ""),
                "category_text_profile": category_profile(text_raw),
                "length_m": nums.get("length_m", ""),
                "displacement_t": nums.get("displacement_t", ""),
                "vls_cells": nums.get("vls_cells", ""),
                "parameter_hint": param_hint_text,
                "gold_evidence_level": gold_prof.get("evidence_level", ""),
                "gold_evidence_score": gold_prof.get("score", ""),
                "gold_explicit_hits": gold_prof.get("explicit_hits", ""),
                "gold_unique_hits": gold_prof.get("unique_hits", ""),
                "gold_shared_hits": gold_prof.get("shared_hits", ""),
                "gold_negative_hits": gold_prof.get("negative_hits", ""),
                "gold_combo_hits": gold_prof.get("combo_hits", ""),
                "pred_evidence_level": pred_prof.get("evidence_level", ""),
                "pred_evidence_score": pred_prof.get("score", ""),
                "pred_explicit_hits": pred_prof.get("explicit_hits", ""),
                "pred_unique_hits": pred_prof.get("unique_hits", ""),
                "pred_shared_hits": pred_prof.get("shared_hits", ""),
                "pred_negative_hits": pred_prof.get("negative_hits", ""),
                "pred_combo_hits": pred_prof.get("combo_hits", ""),
                "suggested_action": "",
            })
            out["suggested_action"] = suggest(out, gold_prof, pred_prof, param_hint_text)
            rows.append(out)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # report
    if args.report:
        from collections import Counter
        c_error = Counter(r.get("diagnosis_error_type", "") for r in rows)
        c_gold = Counter(r.get("gold_evidence_level", "") for r in rows)
        c_pred = Counter(r.get("pred_evidence_level", "") for r in rows)
        c_action = Counter(r.get("suggested_action", "") for r in rows)
        lines = []
        lines.append("# Evidence Profile Diagnosis Report\n")
        lines.append(f"Total errors: {len(rows)}\n")
        lines.append("## Error Types\n")
        for k, v in c_error.most_common():
            lines.append(f"- {k}: {v}")
        lines.append("\n## Gold Evidence Levels\n")
        for k, v in c_gold.most_common():
            lines.append(f"- {k}: {v}")
        lines.append("\n## Pred Evidence Levels\n")
        for k, v in c_pred.most_common():
            lines.append(f"- {k}: {v}")
        lines.append("\n## Suggested Actions\n")
        for k, v in c_action.most_common():
            lines.append(f"- {k}: {v}")
        lines.append("\n## High-value cases\n")
        for r in rows:
            if r.get("suggested_action", "").startswith("可尝试安全恢复") or r.get("diagnosis_error_type") == "unknown_to_known_false_positive":
                lines.append(f"- {r.get('id')}: {r.get('diagnosis_error_type')} | gold={r.get('gold_known_class')} ({r.get('gold_evidence_level')}) | pred={r.get('pred_known_class')} ({r.get('pred_evidence_level')}) | {r.get('suggested_action')}")
        Path(args.report).write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {out_path}")
    if args.report:
        print(f"Wrote: {args.report}")


if __name__ == "__main__":
    main()
