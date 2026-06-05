# -*- coding: utf-8 -*-
"""
replay_match_debug_from_detail.py

用途：
不重新调用 LLM，只复用 predictions_detail.jsonl 中的 observed_attributes，
重放 run_deepseek.hierarchical_class_match()，并把关键调试字段也导出。
这样可以判断错误到底来自：
1. base known_class_result 已经闭集了；
2. v49_1 top_candidate_promotion 提升了；
3. v49_3/v49_7/v49_10 gate 是否触发；
4. open_set/category/final_decision 的最终合成路径。

PowerShell 示例：
.venv\Scripts\python.exe replay_match_debug_from_detail.py `
  --detail batch_results_v17_v2_80\predictions_detail.jsonl `
  --labels sim_text_labels_v2_80.jsonl `
  --output batch_results_v17_v2_80\replay_v49_10_debug.csv
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from run_deepseek import hierarchical_class_match


def read_jsonl_list(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_label_map(path: Path) -> Dict[str, Dict[str, Any]]:
    return {x["id"]: x for x in read_jsonl_list(path)}


def synthesize_prediction(match_result: Dict[str, Any]) -> Dict[str, Any]:
    final_decision = match_result.get("final_decision") or {}
    category_result = match_result.get("category_result") or {}
    known_class_result = match_result.get("known_class_result") or {}
    open_set_result = match_result.get("open_set_result") or {}

    pred_category = (
        final_decision.get("primary_category")
        or category_result.get("label")
        or known_class_result.get("category")
    )
    pred_known_class = (
        final_decision.get("primary_class")
        or known_class_result.get("label")
        or known_class_result.get("ship_class")
    )
    pred_open_set = bool(open_set_result.get("is_unknown", False))
    if final_decision.get("result_type") == "category_unknown":
        pred_open_set = True

    return {
        "pred_category": pred_category,
        "pred_known_class": pred_known_class,
        "pred_open_set": pred_open_set,
        "final_decision": final_decision,
    }


def judge(pred: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, bool]:
    category_correct = pred.get("pred_category") == gold.get("gold_category")
    open_set_correct = bool(pred.get("pred_open_set")) == bool(gold.get("gold_open_set"))
    if gold.get("gold_open_set"):
        known_class_correct = pred.get("pred_known_class") in {None, "", "未知"}
        exact_correct = category_correct and open_set_correct
    else:
        known_class_correct = pred.get("pred_known_class") == gold.get("gold_known_class")
        exact_correct = category_correct and known_class_correct and open_set_correct
    return {
        "category_correct": category_correct,
        "known_class_correct": known_class_correct,
        "open_set_correct": open_set_correct,
        "exact_correct": exact_correct,
    }


def first_candidate(cands: Any) -> Dict[str, Any]:
    if isinstance(cands, list) and cands:
        return cands[0] or {}
    return {}


def jdump(x: Any) -> str:
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--class-data", default="./class_data.txt")
    ap.add_argument("--output", required=True)
    ap.add_argument("--errors-only", action="store_true", help="只导出错误样本")
    args = ap.parse_args()

    detail_rows = read_jsonl_list(Path(args.detail))
    label_map = read_label_map(Path(args.labels))

    out_rows = []
    for row in detail_rows:
        sid = row["id"]
        if sid not in label_map:
            continue
        if not row.get("ok"):
            continue
        observed = row.get("observed_attributes")
        if not isinstance(observed, dict):
            continue

        observed.setdefault("_META", {"raw_text": row.get("input_text", "")})
        observed["_META"]["raw_text"] = row.get("input_text", "")

        match_result = hierarchical_class_match(args.class_data, observed)
        pred = synthesize_prediction(match_result)
        gold = label_map[sid]
        corr = judge(pred, gold)

        final = match_result.get("final_decision") or {}
        cat = match_result.get("category_result") or {}
        known = match_result.get("known_class_result") or {}
        open_set = match_result.get("open_set_result") or {}
        top = first_candidate(match_result.get("known_class_candidates") or match_result.get("top_known_classes") or [])

        promo = match_result.get("v49_1_top_candidate_promotion") or {}
        gate3 = match_result.get("v49_3_open_set_promotion_gate") or {}
        gate7 = match_result.get("v49_7_five_evidence_category_only") or {}
        gate10 = match_result.get("v49_10_no_score_bypass_guard") or {}

        out = {
            "id": sid,
            "gold_category": gold.get("gold_category"),
            "gold_known_class": gold.get("gold_known_class"),
            "gold_open_set": gold.get("gold_open_set"),
            "pred_category": pred.get("pred_category"),
            "pred_known_class": pred.get("pred_known_class"),
            "pred_open_set": pred.get("pred_open_set"),
            "result_type": final.get("result_type"),
            "final_status": final.get("status"),
            "confidence": final.get("confidence"),
            "category_correct": corr["category_correct"],
            "known_class_correct": corr["known_class_correct"],
            "open_set_correct": corr["open_set_correct"],
            "exact_correct": corr["exact_correct"],

            "category_result_label": cat.get("label"),
            "category_result_conf": cat.get("confidence"),
            "known_result_label": known.get("label") or known.get("ship_class"),
            "known_result_category": known.get("category"),
            "known_result_conf": known.get("confidence"),
            "open_set_is_unknown": open_set.get("is_unknown"),
            "open_set_reason": open_set.get("reason"),
            "open_set_unknown_scope": open_set.get("unknown_scope"),

            "top_candidate_label": top.get("label") or top.get("ship_class"),
            "top_candidate_category": top.get("category"),
            "top_candidate_conf": top.get("confidence"),
            "top_candidate_score": top.get("score"),
            "top_candidate_evidence_count": top.get("matched_evidence_count"),
            "top_candidate_conflict_count": top.get("conflict_count"),

            "v49_1_promotion_applied": promo.get("applied"),
            "v49_1_promotion_reason": promo.get("reason"),
            "v49_1_safety_profile": promo.get("safety_profile"),
            "v49_3_gate_applied": gate3.get("applied"),
            "v49_3_gate_action": gate3.get("action"),
            "v49_3_gate_reason": gate3.get("reason"),
            "v49_3_anchor_required": promo.get("v49_3_anchor_required") or gate3.get("v49_3_anchor_required"),
            "v49_3_anchor_passed": promo.get("v49_3_anchor_passed") or gate3.get("v49_3_anchor_passed"),
            "v49_3_anchor_reason": promo.get("v49_3_anchor_reason") or gate3.get("v49_3_anchor_reason"),
            "v49_3_anchor_groups": jdump(promo.get("v49_3_anchor_matched_groups") or gate3.get("v49_3_anchor_matched_groups")),
            "v49_7_category_gate": jdump(gate7),
            "v49_10_guard": jdump(gate10),

            "input_text": row.get("input_text", ""),
        }
        if (not args.errors_only) or (not corr["exact_correct"]):
            out_rows.append(out)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(out_rows[0].keys()) if out_rows else []
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        if fields:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(out_rows)

    total = len(out_rows)
    print(f"导出 {total} 条到: {out_path}")


if __name__ == "__main__":
    main()
