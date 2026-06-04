# -*- coding: utf-8 -*-
r"""
run_all_replay.py

在项目根目录运行：
  .venv\Scripts\python.exe .\run_all_replay.py --tag v49_2

功能：
1. 检查 run_deepseek.py 语法；
2. 依次 replay 原 80 条、原 100 条、anchor 80 条；
3. 输出每组 summary/errors；
4. 汇总 replay_<tag>_metrics.csv 和 replay_<tag>_all_errors.csv。
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any


TASKS = [
    {
        "name": "origin_80",
        "detail": r"batch_results_v17_v2_80\predictions_detail.jsonl",
        "labels": r"sim_text_labels_v2_80.jsonl",
        "out_dir": r"batch_results_v17_v2_80",
    },
    {
        "name": "origin_100",
        "detail": r"batch_results_test_100\predictions_detail.jsonl",
        "labels": r"sim_text_labels_test_100_fixed.jsonl",
        "out_dir": r"batch_results_test_100",
    },
    {
        "name": "anchor_80",
        "detail": r"batch_results_dev_anchor_80\predictions_detail.jsonl",
        "labels": r"sim_text_labels_dev_anchor_80.jsonl",
        "out_dir": r"batch_results_dev_anchor_80",
    },
]


def run_cmd(cmd: List[str]) -> None:
    print("\n> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def bool_true(x: Any) -> bool:
    return str(x).strip().lower() == "true"


def calc_metrics(dataset_name: str, summary_path: Path, error_path: Path) -> Dict[str, Any]:
    rows = read_csv(summary_path)
    total = len(rows)
    if total == 0:
        raise RuntimeError(f"summary 为空：{summary_path}")

    exact = sum(1 for r in rows if bool_true(r.get("exact_correct")))
    cat = sum(1 for r in rows if bool_true(r.get("category_correct")))
    known = sum(1 for r in rows if bool_true(r.get("known_class_correct")))
    open_set = sum(1 for r in rows if bool_true(r.get("open_set_correct")))

    errors = [r for r in rows if not bool_true(r.get("exact_correct"))]
    write_csv(error_path, errors)

    return {
        "dataset": dataset_name,
        "total": total,
        "exact_correct": exact,
        "exact_acc": round(exact / total, 4),
        "category_acc": round(cat / total, 4),
        "known_class_acc": round(known / total, 4),
        "open_set_acc": round(open_set / total, 4),
        "summary_file": str(summary_path),
        "error_file": str(error_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="v49_2", help="输出文件标签，例如 v49_2 / v50 / test")
    parser.add_argument("--skip-pycompile", action="store_true", help="跳过 run_deepseek.py 语法检查")
    args = parser.parse_args()

    root = Path.cwd()
    print(f"项目目录：{root}")
    print(f"Replay tag：{args.tag}")
    print(f"Python：{sys.executable}")

    if not args.skip_pycompile:
        print("\n========== 1. 检查 run_deepseek.py 语法 ==========")
        run_cmd([sys.executable, "-m", "py_compile", "run_deepseek.py"])
        print("语法检查通过。")

    metrics: List[Dict[str, Any]] = []
    all_errors: List[Dict[str, Any]] = []

    print("\n========== 2. 执行三组 replay ==========")
    for task in TASKS:
        name = task["name"]
        detail = root / task["detail"]
        labels = root / task["labels"]
        out_dir = root / task["out_dir"]
        summary = out_dir / f"replay_{args.tag}_summary.csv"
        errors = out_dir / f"replay_{args.tag}_errors.csv"

        print(f"\n--- {name} ---")
        if not detail.exists():
            raise FileNotFoundError(f"找不到 detail 文件：{detail}")
        if not labels.exists():
            raise FileNotFoundError(f"找不到 labels 文件：{labels}")

        run_cmd([
            sys.executable,
            "replay_match_from_detail.py",
            "--detail", str(detail),
            "--labels", str(labels),
            "--output", str(summary),
        ])

        m = calc_metrics(name, summary, errors)
        metrics.append(m)

        err_rows = read_csv(errors)
        for r in err_rows:
            r["dataset"] = name
            all_errors.append(r)

        print(f"summary：{summary}")
        print(f"errors ：{errors}")
        print(f"exact_acc={m['exact_acc']}, category_acc={m['category_acc']}, known_class_acc={m['known_class_acc']}, open_set_acc={m['open_set_acc']}")

    print("\n========== 3. 汇总输出 ==========")
    metrics_path = root / f"replay_{args.tag}_metrics.csv"
    all_errors_path = root / f"replay_{args.tag}_all_errors.csv"

    write_csv(metrics_path, metrics)
    write_csv(all_errors_path, all_errors)

    print(f"总指标：{metrics_path}")
    print(f"全部错误：{all_errors_path}")

    print("\n指标预览：")
    for m in metrics:
        print(
            f"{m['dataset']:>10} | total={m['total']:>3} | "
            f"exact={m['exact_acc']:.4f} | category={m['category_acc']:.4f} | "
            f"known={m['known_class_acc']:.4f} | open={m['open_set_acc']:.4f}"
        )

    print("\n全部完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
