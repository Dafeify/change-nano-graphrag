# -*- coding: utf-8 -*-
"""
batch_test_sim_text.py

批量测试仿真文本数据集：
1. 只把 sim_text_inputs_72.jsonl 里的 input_text 输入模型；
2. sim_text_labels_72.jsonl 只用于最后评估，不参与模型推理；
3. 复用 run_deepseek.py 中的 direct_text_parse_v2 / hierarchical_class_match 等函数；
4. 输出逐条预测结果、CSV简表和总体评估指标。

推荐放置位置：
与 run_deepseek.py、class_data.txt、schema_config.py、
sim_text_inputs_72.jsonl、sim_text_labels_72.jsonl 放在同一目录。

运行示例：
python batch_test_sim_text.py

先测试前 5 条：
python batch_test_sim_text.py --limit 5

断点续跑：
python batch_test_sim_text.py --resume
"""

import argparse
import asyncio
import csv
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

try:
    from run_deepseek import (
        build_class_graph_from_class_data,
        print_graph_summary,
        direct_text_parse_v2,
        extract_json_object,
        normalize_observed_attributes,
        validate_observed_schema,
        hierarchical_class_match,
    )
except Exception as e:
    raise RuntimeError(
        "无法从 run_deepseek.py 导入必要函数。请确认：\n"
        "1. batch_test_sim_text.py 与 run_deepseek.py 在同一目录；\n"
        "2. 你使用的是新版 run_deepseek.py；\n"
        "3. run_deepseek.py 中保留了 direct_text_parse_v2、hierarchical_class_match 等函数。"
    ) from e


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON: {e}") from e
    return data


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_results(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    result = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if "id" in item:
                result[item["id"]] = item
    return result


def synthesize_final_decision(match_result: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(match_result, dict) and isinstance(match_result.get("final_decision"), dict):
        return match_result["final_decision"]

    category_result = match_result.get("category_result") or {}
    known_class_result = match_result.get("known_class_result")
    open_set_result = match_result.get("open_set_result") or {}

    if known_class_result:
        return {
            "result_type": "known_class",
            "primary_category": known_class_result.get("category") or category_result.get("label"),
            "primary_class": known_class_result.get("label"),
            "confidence": known_class_result.get("confidence", category_result.get("confidence", 0.0)),
            "status": "single_best",
            "message": f"最终判定：{known_class_result.get('category')} / {known_class_result.get('label')}。",
        }

    if open_set_result.get("is_unknown"):
        return {
            "result_type": "category_unknown",
            "primary_category": category_result.get("label"),
            "primary_class": None,
            "confidence": category_result.get("confidence", 0.0),
            "status": "open_set",
            "message": f"最终判定：{open_set_result.get('unknown_scope')}。",
        }

    return {
        "result_type": "uncertain",
        "primary_category": category_result.get("label"),
        "primary_class": None,
        "confidence": category_result.get("confidence", 0.0),
        "status": "uncertain",
        "message": category_result.get("reason", "无法稳定判断。"),
    }


def extract_prediction(match_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    从 match_result 中提取预测结果。
    优先读取 final_decision；如果 final_decision 不完整，
    回退读取 category_result / known_class_result，避免 pred_known_class 为空。
    """
    final_decision = synthesize_final_decision(match_result)
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
    pred_unknown_scope = open_set_result.get("unknown_scope")

    if final_decision.get("result_type") == "category_unknown":
        pred_open_set = True
        pred_unknown_scope = pred_unknown_scope or final_decision.get("message")

    return {
        "pred_category": pred_category,
        "pred_known_class": pred_known_class,
        "pred_open_set": pred_open_set,
        "pred_unknown_scope": pred_unknown_scope,
        "final_decision": final_decision,
    }


def judge_correctness(pred: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, bool]:
    gold_category = gold.get("gold_category")
    gold_known_class = gold.get("gold_known_class")
    gold_open_set = bool(gold.get("gold_open_set", False))
    gold_unknown_scope = gold.get("gold_unknown_scope")

    pred_category = pred.get("pred_category")
    pred_known_class = pred.get("pred_known_class")
    pred_open_set = bool(pred.get("pred_open_set", False))
    pred_unknown_scope = pred.get("pred_unknown_scope")

    category_correct = pred_category == gold_category
    open_set_correct = pred_open_set == gold_open_set

    if gold_open_set:
        known_class_correct = pred_known_class in {None, "", "未知"}
        unknown_scope_correct = pred_unknown_scope == gold_unknown_scope
        exact_correct = category_correct and open_set_correct
    else:
        known_class_correct = pred_known_class == gold_known_class
        unknown_scope_correct = True
        exact_correct = category_correct and known_class_correct and open_set_correct

    return {
        "category_correct": category_correct,
        "known_class_correct": known_class_correct,
        "open_set_correct": open_set_correct,
        "unknown_scope_correct": unknown_scope_correct,
        "exact_correct": exact_correct,
    }


async def run_one_sample(
    sample: Dict[str, Any],
    gold: Dict[str, Any],
    class_data_path: str,
    sleep_seconds: float = 0.0,
) -> Dict[str, Any]:
    # 注意：这里只把 input_text 传给模型；gold 只用于模型输出后的评估。
    sample_id = sample["id"]
    input_text = sample["input_text"]

    row: Dict[str, Any] = {
        "id": sample_id,
        "source_type": sample.get("source_type"),
        "description_level": sample.get("description_level"),
        "noise_level": sample.get("noise_level"),
        "input_text": input_text,
        "gold": gold,
        "ok": False,
        "error": None,
    }

    try:
        parse_text = await direct_text_parse_v2(input_text)
        parsed_obj = extract_json_object(parse_text)
        observed = normalize_observed_attributes(parsed_obj)
        # 保存原始输入文本给层级匹配模块使用。
        # 例如：文本中出现“与已知类不一致/不像已知类”时，开放集判断需要这个信息。
        observed["_META"] = {"raw_text": input_text}
        schema_errors = validate_observed_schema({k: v for k, v in observed.items() if k != "_META"})

        match_result = hierarchical_class_match(class_data_path, observed)
        pred = extract_prediction(match_result)
        correctness = judge_correctness(pred, gold)

        row.update({
            "ok": True,
            "parse_text": parse_text,
            "observed_attributes": observed,
            "schema_errors": schema_errors,
            "match_result": match_result,
            "prediction": pred,
            "correctness": correctness,
        })

    except Exception as e:
        row.update({
            "ok": False,
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            },
        })

    if sleep_seconds > 0:
        await asyncio.sleep(sleep_seconds)

    return row


def safe_div(a: int, b: int) -> float:
    return round(a / b, 4) if b else 0.0


def compute_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_rows = [r for r in rows if r.get("ok")]
    failed_rows = [r for r in rows if not r.get("ok")]
    valid_n = len(valid_rows)

    def count(metric: str) -> int:
        return sum(1 for r in valid_rows if r.get("correctness", {}).get(metric))

    known_rows = [r for r in valid_rows if not bool(r.get("gold", {}).get("gold_open_set", False))]
    unknown_rows = [r for r in valid_rows if bool(r.get("gold", {}).get("gold_open_set", False))]

    def count_in(rows_subset: List[Dict[str, Any]], metric: str) -> int:
        return sum(1 for r in rows_subset if r.get("correctness", {}).get(metric))

    def group_metric(key: str) -> Dict[str, Any]:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in valid_rows:
            groups.setdefault(str(r.get(key, "UNKNOWN")), []).append(r)
        result = {}
        for g, items in sorted(groups.items()):
            result[g] = {
                "total": len(items),
                "category_acc": safe_div(count_in(items, "category_correct"), len(items)),
                "exact_acc": safe_div(count_in(items, "exact_correct"), len(items)),
                "open_set_acc": safe_div(count_in(items, "open_set_correct"), len(items)),
            }
        return result

    return {
        "total": len(rows),
        "valid": valid_n,
        "failed": len(failed_rows),
        "overall": {
            "category_acc": safe_div(count("category_correct"), valid_n),
            "known_class_acc_or_unknown_no_class": safe_div(count("known_class_correct"), valid_n),
            "open_set_acc": safe_div(count("open_set_correct"), valid_n),
            "exact_acc": safe_div(count("exact_correct"), valid_n),
        },
        "known_samples": {
            "total": len(known_rows),
            "category_acc": safe_div(count_in(known_rows, "category_correct"), len(known_rows)),
            "known_class_acc": safe_div(count_in(known_rows, "known_class_correct"), len(known_rows)),
            "exact_acc": safe_div(count_in(known_rows, "exact_correct"), len(known_rows)),
        },
        "unknown_samples": {
            "total": len(unknown_rows),
            "category_acc": safe_div(count_in(unknown_rows, "category_correct"), len(unknown_rows)),
            "open_set_recall": safe_div(count_in(unknown_rows, "open_set_correct"), len(unknown_rows)),
            "exact_acc": safe_div(count_in(unknown_rows, "exact_correct"), len(unknown_rows)),
        },
        "by_source_type": group_metric("source_type"),
        "by_description_level": group_metric("description_level"),
        "by_noise_level": group_metric("noise_level"),
        "failed_ids": [r["id"] for r in failed_rows],
    }


def write_csv_report(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id", "source_type", "description_level", "noise_level",
        "gold_category", "gold_known_class", "gold_open_set", "gold_unknown_scope",
        "pred_category", "pred_known_class", "pred_open_set", "pred_unknown_scope",
        "decision_confidence", "result_type",
        "category_correct", "known_class_correct", "open_set_correct", "exact_correct",
        "message", "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            gold = r.get("gold", {})
            pred = r.get("prediction", {}) if isinstance(r.get("prediction"), dict) else {}
            fd = pred.get("final_decision", {}) if isinstance(pred.get("final_decision"), dict) else {}
            corr = r.get("correctness", {})
            writer.writerow({
                "id": r.get("id"),
                "source_type": r.get("source_type"),
                "description_level": r.get("description_level"),
                "noise_level": r.get("noise_level"),
                "gold_category": gold.get("gold_category"),
                "gold_known_class": gold.get("gold_known_class"),
                "gold_open_set": gold.get("gold_open_set"),
                "gold_unknown_scope": gold.get("gold_unknown_scope"),
                "pred_category": pred.get("pred_category"),
                "pred_known_class": pred.get("pred_known_class"),
                "pred_open_set": pred.get("pred_open_set"),
                "pred_unknown_scope": pred.get("pred_unknown_scope"),
                "decision_confidence": fd.get("confidence"),
                "result_type": fd.get("result_type"),
                "category_correct": corr.get("category_correct"),
                "known_class_correct": corr.get("known_class_correct"),
                "open_set_correct": corr.get("open_set_correct"),
                "exact_correct": corr.get("exact_correct"),
                "message": fd.get("message"),
                "error": r.get("error", {}).get("message") if r.get("error") else "",
            })


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default="./sim_text_inputs_72.jsonl", help="仿真输入文件，只包含 input_text 等字段")
    parser.add_argument("--labels", default="./sim_text_labels_72.jsonl", help="正确答案标签文件，只用于评估")
    parser.add_argument("--class-data", default="./class_data.txt", help="class_data.txt 路径")
    parser.add_argument("--working-dir", default="./class_index", help="GraphML 输出目录")
    parser.add_argument("--output-dir", default="./batch_results", help="批量测试输出目录")
    parser.add_argument("--limit", type=int, default=0, help="只测试前 N 条；0 表示全部")
    parser.add_argument("--start", type=int, default=0, help="从第几个样本开始，默认 0")
    parser.add_argument("--sleep", type=float, default=0.0, help="每条之间暂停秒数，防止 API 限流")
    parser.add_argument("--rebuild-graph", action="store_true", help="重新构建 GraphML")
    parser.add_argument("--resume", action="store_true", help="断点续跑：跳过已经成功跑过的 id")
    args = parser.parse_args()

    inputs_path = Path(args.inputs)
    labels_path = Path(args.labels)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_path = output_dir / "predictions_detail.jsonl"
    csv_path = output_dir / "predictions_summary.csv"
    summary_path = output_dir / "metrics_summary.json"

    graphml_file = Path(args.working_dir) / "graph_chunk_entity_relation.graphml"
    if args.rebuild_graph or not graphml_file.exists():
        print("[批量测试] 构建/重建 GraphML...")
        build_class_graph_from_class_data(args.working_dir, args.class_data, rebuild=args.rebuild_graph)
    else:
        print(f"[批量测试] 复用已有 GraphML: {graphml_file}")
    print_graph_summary(args.working_dir)

    inputs = read_jsonl(inputs_path)
    labels = read_jsonl(labels_path)
    label_map = {x["id"]: x for x in labels}

    if args.start:
        inputs = inputs[args.start:]
    if args.limit and args.limit > 0:
        inputs = inputs[:args.limit]

    print(f"[批量测试] 待测试样本数: {len(inputs)}")
    print(f"[批量测试] 输入文件: {inputs_path}")
    print(f"[批量测试] 标签文件: {labels_path}")
    print(f"[批量测试] 输出目录: {output_dir}")

    existing = load_existing_results(detail_path) if args.resume else {}
    rows: List[Dict[str, Any]] = []
    if args.resume and existing:
        print(f"[批量测试] 检测到已有结果 {len(existing)} 条，将跳过已完成样本。")
        rows.extend(existing.values())
    elif detail_path.exists():
        detail_path.unlink()

    for idx, sample in enumerate(inputs, start=1):
        sample_id = sample["id"]
        if args.resume and sample_id in existing and existing[sample_id].get("ok"):
            print(f"[{idx}/{len(inputs)}] 跳过已完成: {sample_id}")
            continue

        gold = label_map.get(sample_id)
        if not gold:
            raise KeyError(f"标签文件中找不到 id={sample_id} 的正确答案。")

        print(f"\n[{idx}/{len(inputs)}] 开始测试: {sample_id}")
        t0 = time.time()
        row = await run_one_sample(sample, gold, args.class_data, sleep_seconds=args.sleep)
        cost = time.time() - t0
        row["runtime_seconds"] = round(cost, 3)

        pred = row.get("prediction", {}) if isinstance(row.get("prediction"), dict) else {}
        corr = row.get("correctness", {}) if isinstance(row.get("correctness"), dict) else {}
        if row.get("ok"):
            print(
                f"  预测: category={pred.get('pred_category')} | "
                f"class={pred.get('pred_known_class')} | "
                f"open_set={pred.get('pred_open_set')} | "
                f"exact={corr.get('exact_correct')} | "
                f"耗时={cost:.1f}s"
            )
        else:
            err = row.get("error", {})
            print(f"  失败: {err.get('type')} - {err.get('message')}")

        append_jsonl(detail_path, row)
        rows = [r for r in rows if r.get("id") != sample_id]
        rows.append(row)

    order = {x["id"]: i for i, x in enumerate(read_jsonl(inputs_path))}
    rows.sort(key=lambda r: order.get(r.get("id"), 999999))

    summary = compute_summary(rows)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_csv_report(csv_path, rows)

    print("\n" + "=" * 60)
    print("【批量测试完成】")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n详细结果: {detail_path}")
    print(f"CSV简表: {csv_path}")
    print(f"指标汇总: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
