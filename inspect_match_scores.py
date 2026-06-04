# -*- coding: utf-8 -*-
"""
inspect_match_scores.py

用途：诊断某些样本为什么被判错：
1. 从 predictions_detail.jsonl 找到指定 id 的详细结果；
2. 从 replay/predictions_summary.csv 找到 gold/pred 标签；
3. 输出每条样本的 top category、top known class、matched_evidence、conflict_evidence、observed_attributes 摘要。

用法：
python inspect_match_scores.py --detail batch_results_v17_v2_80\predictions_detail.jsonl \
  --summary batch_results_v17_v2_80\replay_v38_summary.csv \
  --ids known_arleigh_burke_002 known_wasp_007 \
  --out debug_score_inspect.csv

如果不传 --ids，则默认分析 summary 里 exact_correct != True 的所有样本。
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = str(obj.get("id", ""))
            if sid:
                rows[sid] = obj
    return rows


def read_csv(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {str(r.get("id", "")): r for r in csv.DictReader(f) if r.get("id")}


def compact_json(x: Any, max_len: int = 900) -> str:
    s = json.dumps(x, ensure_ascii=False)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def pick_first(obj: Dict[str, Any], keys: List[str], default=None):
    cur = obj
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            return cur[k]
    return default


def get_final_result(d: Dict[str, Any]) -> Dict[str, Any]:
    for key in ["final_decision", "final_result", "hierarchical_result", "result"]:
        v = d.get(key)
        if isinstance(v, dict):
            return v
    return {}


def get_category_candidates(d: Dict[str, Any]) -> Any:
    for path in [
        ["category_candidates"],
        ["hierarchical_result", "category_candidates"],
        ["result", "category_candidates"],
        ["final_decision", "alternatives", "top_categories"],
    ]:
        cur = d
        ok = True
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                ok = False
                break
        if ok:
            return cur
    return []


def get_class_candidates(d: Dict[str, Any]) -> Any:
    for path in [
        ["class_results"],
        ["known_class_candidates"],
        ["hierarchical_result", "class_results"],
        ["hierarchical_result", "known_class_candidates"],
        ["result", "class_results"],
        ["final_decision", "alternatives", "top_known_classes"],
    ]:
        cur = d
        ok = True
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                ok = False
                break
        if ok:
            return cur
    return []


def evidence_summary(cands: Any, topn: int = 3) -> str:
    if not isinstance(cands, list):
        return compact_json(cands)
    out = []
    for c in cands[:topn]:
        if not isinstance(c, dict):
            out.append(str(c))
            continue
        item = {
            "label": c.get("label") or c.get("ship_class") or c.get("category"),
            "category": c.get("category"),
            "confidence": c.get("confidence"),
            "score": c.get("score"),
            "matched_count": len(c.get("matched_evidence", []) or []),
            "conflict_count": len(c.get("conflict_evidence", []) or []),
            "matched_evidence": (c.get("matched_evidence", []) or [])[:6],
            "conflict_evidence": (c.get("conflict_evidence", []) or [])[:6],
        }
        out.append(item)
    return compact_json(out, max_len=2000)


def flat_slots(obs: Dict[str, Any]) -> str:
    keep = []
    for group, slots in (obs or {}).items():
        if str(group).startswith("_") or not isinstance(slots, dict):
            continue
        for slot, val in slots.items():
            if val in (None, "", "未知", "不确定", "未提及", []):
                continue
            keep.append(f"{group}.{slot}={val}")
    return " | ".join(keep[:80])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", nargs="*", default=None)
    args = ap.parse_args()

    details = read_jsonl(Path(args.detail))
    summary = read_csv(Path(args.summary))

    if args.ids:
        ids = args.ids
    else:
        ids = [sid for sid, r in summary.items() if str(r.get("exact_correct", "")).lower() != "true"]

    fields = [
        "id", "gold_category", "gold_known_class", "gold_open_set",
        "pred_category", "pred_known_class", "pred_open_set", "exact_correct",
        "input_text", "observed_non_unknown_slots",
        "final_decision", "category_candidates", "class_candidates",
    ]
    with Path(args.out).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sid in ids:
            s = summary.get(sid, {})
            d = details.get(sid, {})
            obs = d.get("observed_attributes", {})
            w.writerow({
                "id": sid,
                "gold_category": s.get("gold_category", ""),
                "gold_known_class": s.get("gold_known_class", ""),
                "gold_open_set": s.get("gold_open_set", ""),
                "pred_category": s.get("pred_category", ""),
                "pred_known_class": s.get("pred_known_class", ""),
                "pred_open_set": s.get("pred_open_set", ""),
                "exact_correct": s.get("exact_correct", ""),
                "input_text": d.get("input_text", ""),
                "observed_non_unknown_slots": flat_slots(obs),
                "final_decision": compact_json(get_final_result(d), max_len=1200),
                "category_candidates": evidence_summary(get_category_candidates(d), topn=5),
                "class_candidates": evidence_summary(get_class_candidates(d), topn=5),
            })
    print(f"输出完成: {args.out}")


if __name__ == "__main__":
    main()
