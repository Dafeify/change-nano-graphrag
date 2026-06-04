# -*- coding: utf-8 -*-
"""
inspect_selected_cases.py

用途：
从 predictions_detail.jsonl 和 replay/predictions_summary.csv 中抽取指定样本的：
1. 原始输入文本
2. gold / pred 结果
3. observed_attributes 非空卡槽
4. 大类候选、已知舰级候选、matched evidence、conflict evidence

运行示例：
.venv\Scripts\python.exe inspect_selected_cases.py ^
  --detail batch_results_v17_v2_80\predictions_detail.jsonl ^
  --summary batch_results_v17_v2_80\replay_v38_summary.csv ^
  --ids inspect_ids_arleigh.txt ^
  --out-prefix inspect_arleigh_v38
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Iterable, Tuple


UNKNOWN_VALUES = {"", "未知", "不确定", "未提及", "N/A", "None", "null", None}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {r.get("id", ""): r for r in rows if r.get("id")}


def read_ids(path: Path) -> List[str]:
    ids = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                ids.append(s)
    return ids


def compact_json(x: Any, max_len: int = 900) -> str:
    s = json.dumps(x, ensure_ascii=False, separators=(",", ":"))
    return s if len(s) <= max_len else s[:max_len] + "..."


def is_unknown(v: Any) -> bool:
    if isinstance(v, list):
        return len(v) == 0 or all(is_unknown(x) for x in v)
    return v in UNKNOWN_VALUES or str(v).strip() in UNKNOWN_VALUES


def flatten_observed(obs: Any) -> Dict[str, Any]:
    """把 observed_attributes 展平成 GROUP.slot -> value，只保留非空/非未知字段。"""
    out = {}
    if not isinstance(obs, dict):
        return out

    for group, obj in obs.items():
        if str(group).startswith("_"):
            continue
        if not isinstance(obj, dict):
            continue
        for slot, val in obj.items():
            if not is_unknown(val):
                out[f"{group}.{slot}"] = val
    return out


def find_first_key(obj: Any, wanted_keys: Iterable[str]) -> Any:
    wanted = set(wanted_keys)
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in wanted:
                return v
        for v in obj.values():
            found = find_first_key(v, wanted)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_first_key(item, wanted_keys)
            if found is not None:
                return found
    return None


def find_candidate_lists(obj: Any) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """递归找候选列表：元素中包含 label/ship_class/category/confidence/score 等。"""
    found = []

    def walk(x: Any, path: str):
        if isinstance(x, list) and x and all(isinstance(i, dict) for i in x):
            keys = set()
            for i in x[:3]:
                keys.update(i.keys())
            if keys & {"label", "ship_class", "category", "confidence", "score"}:
                # 过滤 evidence 这种列表，优先候选列表
                if not (keys <= {"slot", "group", "input_value", "prototype_value", "score", "reason", "base_weight", "specificity_factor", "feature_df", "feature_total"}):
                    found.append((path, x))
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(x, list):
            for idx, item in enumerate(x):
                walk(item, f"{path}[{idx}]")

    walk(obj, "")
    return found


def candidate_name(c: Dict[str, Any]) -> str:
    return str(c.get("ship_class") or c.get("label") or c.get("primary_class") or c.get("name") or "")


def candidate_category(c: Dict[str, Any]) -> str:
    return str(c.get("category") or c.get("primary_category") or "")


def candidate_conf(c: Dict[str, Any]) -> Any:
    return c.get("confidence", c.get("score", ""))


def summarize_candidates(cands: List[Dict[str, Any]], n: int = 5) -> str:
    items = []
    for c in cands[:n]:
        name = candidate_name(c)
        cat = candidate_category(c)
        conf = candidate_conf(c)
        score = c.get("score", "")
        evn = len(c.get("matched_evidence", []) or c.get("evidence", []) or [])
        cfn = len(c.get("conflict_evidence", []) or c.get("conflicts", []) or [])
        items.append({
            "name": name,
            "category": cat,
            "confidence_or_score": conf,
            "score": score,
            "evidence_count": evn,
            "conflict_count": cfn,
        })
    return compact_json(items, max_len=1200)


def find_candidate_for_class(obj: Any, class_name: str) -> Dict[str, Any]:
    lists = find_candidate_lists(obj)
    for _path, cands in lists:
        for c in cands:
            if class_name in candidate_name(c):
                return c
    return {}


def short_evidence(c: Dict[str, Any], key: str, n: int = 8) -> str:
    arr = c.get(key, [])
    if not isinstance(arr, list):
        return ""
    brief = []
    for e in arr[:n]:
        if isinstance(e, dict):
            brief.append({
                "slot": e.get("slot"),
                "input": e.get("input_value"),
                "proto": e.get("prototype_value"),
                "score": e.get("score", e.get("penalty", "")),
                "reason": e.get("reason", ""),
            })
    return compact_json(brief, max_len=1200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", required=True, help="predictions_detail.jsonl 路径")
    ap.add_argument("--summary", required=False, default="", help="replay_vxx_summary.csv 或 predictions_summary.csv 路径")
    ap.add_argument("--ids", required=True, help="待检查样本 id 列表 txt，每行一个 id")
    ap.add_argument("--out-prefix", required=True, help="输出文件前缀，例如 inspect_arleigh_v38")
    args = ap.parse_args()

    detail_path = Path(args.detail)
    summary_path = Path(args.summary) if args.summary else None
    ids_path = Path(args.ids)
    out_prefix = Path(args.out_prefix)

    ids = read_ids(ids_path)
    details = {r.get("id", ""): r for r in read_jsonl(detail_path)}
    summaries = read_csv(summary_path) if summary_path else {}

    selected = []
    for sid in ids:
        d = details.get(sid)
        if d is None:
            print(f"[WARN] detail not found: {sid}")
            continue
        selected.append(d)

    jsonl_out = out_prefix.with_suffix(".jsonl")
    csv_out = out_prefix.with_suffix(".csv")
    txt_out = out_prefix.with_suffix(".txt")

    with jsonl_out.open("w", encoding="utf-8") as f:
        for d in selected:
            sid = d.get("id", "")
            pack = {
                "summary": summaries.get(sid, {}),
                "detail": d,
            }
            f.write(json.dumps(pack, ensure_ascii=False) + "\n")

    fieldnames = [
        "id",
        "gold_category", "gold_known_class", "gold_open_set",
        "pred_category", "pred_known_class", "pred_open_set", "exact_correct",
        "input_text",
        "non_empty_observed_slots",
        "category_result",
        "known_class_result",
        "open_set_result",
        "candidate_list_paths",
        "top_candidates_1",
        "top_candidates_2",
        "arleigh_candidate",
        "arleigh_matched_evidence",
        "arleigh_conflict_evidence",
    ]

    with csv_out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for d in selected:
            sid = d.get("id", "")
            s = summaries.get(sid, {})
            obs = d.get("observed_attributes") or find_first_key(d, ["observed_attributes"]) or {}
            non_empty = flatten_observed(obs)

            candidate_lists = find_candidate_lists(d)
            cand_paths = [p for p, _ in candidate_lists]

            top1 = summarize_candidates(candidate_lists[0][1], 6) if len(candidate_lists) >= 1 else ""
            top2 = summarize_candidates(candidate_lists[1][1], 6) if len(candidate_lists) >= 2 else ""

            arleigh = find_candidate_for_class(d, "阿利")
            writer.writerow({
                "id": sid,
                "gold_category": s.get("gold_category", ""),
                "gold_known_class": s.get("gold_known_class", ""),
                "gold_open_set": s.get("gold_open_set", ""),
                "pred_category": s.get("pred_category", ""),
                "pred_known_class": s.get("pred_known_class", ""),
                "pred_open_set": s.get("pred_open_set", ""),
                "exact_correct": s.get("exact_correct", ""),
                "input_text": d.get("input_text", ""),
                "non_empty_observed_slots": compact_json(non_empty, max_len=2500),
                "category_result": compact_json(find_first_key(d, ["category_result"]) or {}, max_len=1000),
                "known_class_result": compact_json(find_first_key(d, ["known_class_result"]) or {}, max_len=1000),
                "open_set_result": compact_json(find_first_key(d, ["open_set_result"]) or {}, max_len=1000),
                "candidate_list_paths": " | ".join(cand_paths[:10]),
                "top_candidates_1": top1,
                "top_candidates_2": top2,
                "arleigh_candidate": compact_json(arleigh, max_len=1500) if arleigh else "",
                "arleigh_matched_evidence": short_evidence(arleigh, "matched_evidence") if arleigh else "",
                "arleigh_conflict_evidence": short_evidence(arleigh, "conflict_evidence") if arleigh else "",
            })

    with txt_out.open("w", encoding="utf-8") as f:
        for d in selected:
            sid = d.get("id", "")
            s = summaries.get(sid, {})
            obs = d.get("observed_attributes") or find_first_key(d, ["observed_attributes"]) or {}
            non_empty = flatten_observed(obs)
            f.write("=" * 100 + "\n")
            f.write(f"ID: {sid}\n")
            f.write(f"GOLD/PRED: {json.dumps(s, ensure_ascii=False)}\n")
            f.write(f"INPUT: {d.get('input_text', '')}\n\n")
            f.write("NON_EMPTY_OBSERVED_SLOTS:\n")
            f.write(json.dumps(non_empty, ensure_ascii=False, indent=2) + "\n\n")
            f.write("CANDIDATE_LISTS:\n")
            for path, cands in find_candidate_lists(d)[:6]:
                f.write(f"- {path}:\n")
                f.write(summarize_candidates(cands, 8) + "\n")
            f.write("\n")

    print(f"已输出: {jsonl_out}")
    print(f"已输出: {csv_out}")
    print(f"已输出: {txt_out}")


if __name__ == "__main__":
    main()
