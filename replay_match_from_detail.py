# -*- coding: utf-8 -*-
"""
replay_match_from_detail.py

用途：不重新调用 LLM，只复用 batch_results/predictions_detail.jsonl 中已经解析好的 observed_attributes，
重新运行 hierarchical_class_match() 和评估指标。

适合调 run_deepseek.py 里的匹配逻辑、开放集阈值、权重时使用。速度通常是秒级。

运行：
.venv\Scripts\python.exe replay_match_from_detail.py

只重放错误集：
.venv\Scripts\python.exe replay_match_from_detail.py --ids focus_error_ids.txt
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
    """
    从 match_result 中提取最终预测。
    注意：优先读 final_decision，但如果 final_decision 字段不完整，
    必须回退读取 category_result / known_class_result。
    这样可以避免 run_deepseek 已经补全 known_class_result，
    但 replay_summary.csv 里 pred_known_class 仍然为空。
    """
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

    pred_unknown_scope = open_set_result.get("unknown_scope")

    return {
        "pred_category": pred_category,
        "pred_known_class": pred_known_class,
        "pred_open_set": pred_open_set,
        "pred_unknown_scope": pred_unknown_scope,
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


def div(a,b):
    return round(a/b,4) if b else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", default="./batch_results/predictions_detail.jsonl")
    ap.add_argument("--labels", default="./sim_text_labels_72.jsonl")
    ap.add_argument("--class-data", default="./class_data.txt")
    ap.add_argument("--ids", default="", help="只重放指定ID列表，每行一个ID")
    ap.add_argument("--output", default="./batch_results/replay_summary.csv")
    args = ap.parse_args()

    detail_rows = read_jsonl_list(Path(args.detail))
    label_map = read_label_map(Path(args.labels))
    selected_ids = None
    if args.ids:
        selected_ids = {x.strip() for x in Path(args.ids).read_text(encoding="utf-8").splitlines() if x.strip()}

    out_rows = []
    for row in detail_rows:
        sid = row["id"]
        if selected_ids and sid not in selected_ids:
            continue
        if not row.get("ok"):
            continue
        observed = row.get("observed_attributes")
        if not isinstance(observed, dict):
            continue
        # 确保旧 detail 里没有 _META 时也补上原始文本，便于新版开放集逻辑使用。
        observed.setdefault("_META", {"raw_text": row.get("input_text", "")})
        observed["_META"]["raw_text"] = row.get("input_text", "")

        match_result = hierarchical_class_match(args.class_data, observed)
        pred = synthesize_prediction(match_result)
        gold = label_map[sid]
        corr = judge(pred, gold)
        out_rows.append({
            "id": sid,
            "gold_category": gold.get("gold_category"),
            "gold_known_class": gold.get("gold_known_class"),
            "gold_open_set": gold.get("gold_open_set"),
            "pred_category": pred.get("pred_category"),
            "pred_known_class": pred.get("pred_known_class"),
            "pred_open_set": pred.get("pred_open_set"),
            "result_type": pred.get("final_decision",{}).get("result_type"),
            "confidence": pred.get("final_decision",{}).get("confidence"),
            "category_correct": corr["category_correct"],
            "known_class_correct": corr["known_class_correct"],
            "open_set_correct": corr["open_set_correct"],
            "exact_correct": corr["exact_correct"],
        })

    n=len(out_rows)
    summary={
        "total": n,
        "category_acc": div(sum(r["category_correct"] for r in out_rows), n),
        "known_class_acc_or_unknown_no_class": div(sum(r["known_class_correct"] for r in out_rows), n),
        "open_set_acc": div(sum(r["open_set_correct"] for r in out_rows), n),
        "exact_acc": div(sum(r["exact_correct"] for r in out_rows), n),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields=list(out_rows[0].keys()) if out_rows else []
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        if fields:
            w=csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(out_rows)
    print(f"重放结果: {out_path}")

if __name__ == "__main__":
    main()
