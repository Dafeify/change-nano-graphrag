# -*- coding: utf-8 -*-
r"""
ship_file_batch_inference_baseline.py

基于当前 run_deepseek.py 的无 RAG 批量推理入口。

功能：
1. 接收一组文本文件路径；
2. 逐个读取文本文件；
3. 调用 run_deepseek.py 完成舰船大类和小类识别；
4. 大类置信度使用 category_result.confidence；
5. 已知小类置信度使用最终小类对应的原始候选置信度；
6. 输出精简 JSON，仅保留：
   - file_path
   - status
   - category_result
   - category_confidence
   - small_class_result
   - small_class_confidence

说明：
- 不使用 RAG；
- 不涉及训练；
- 支持本地绝对路径、相对路径、Windows UNC 共享路径；
- paths-file 支持 JSON 数组、包含 paths/files 数组的 JSON 对象，
  或每行一个路径的普通文本。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from run_deepseek import (
    direct_text_parse_v2,
    extract_json_object,
    normalize_observed_attributes,
    validate_observed_schema,
    hierarchical_class_match,
)


SUPPORTED_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")


def decode_text(data: bytes, source: str) -> Tuple[str, str]:
    """依次尝试常见中文文本编码。"""
    last_error: Optional[Exception] = None

    for encoding in SUPPORTED_ENCODINGS:
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeError(
        f"无法识别文本编码：{source}。"
        f"已尝试：{', '.join(SUPPORTED_ENCODINGS)}"
    ) from last_error


def read_text_file(path: Path) -> Tuple[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")

    if not path.is_file():
        raise ValueError(f"路径不是普通文件：{path}")

    return decode_text(path.read_bytes(), str(path))


def normalize_path(raw_path: str, base_dir: Optional[Path] = None) -> Path:
    path = Path(str(raw_path).strip())

    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path

    return path


def normalize_path_item(item: Any, base_dir: Optional[Path]) -> Path:
    """
    支持以下 JSON 元素：
    "D:\\data\\001.txt"
    {"path": "D:\\data\\001.txt"}
    {"file_path": "D:\\data\\001.txt"}
    """
    if isinstance(item, str):
        return normalize_path(item, base_dir)

    if not isinstance(item, dict):
        raise ValueError(
            "路径清单中的元素必须是字符串或对象，"
            f"实际类型：{type(item).__name__}"
        )

    raw_path = item.get("path") or item.get("file_path") or item.get("source")
    if not raw_path:
        raise ValueError(f"路径对象缺少 path/file_path/source 字段：{item}")

    return normalize_path(str(raw_path), base_dir)


def load_paths_file(paths_file: Path) -> List[Path]:
    """
    支持：
    1. JSON 数组：
       ["D:\\a.txt", "D:\\b.txt"]

    2. JSON 对象：
       {"paths": ["D:\\a.txt", "D:\\b.txt"]}
       {"files": [{"path": "D:\\a.txt"}]}

    3. 普通文本：
       每行一个路径
    """
    text, _encoding = read_text_file(paths_file)
    stripped = text.strip()

    if not stripped:
        return []

    try:
        payload = json.loads(stripped)

        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict) and isinstance(payload.get("paths"), list):
            items = payload["paths"]
        elif isinstance(payload, dict) and isinstance(payload.get("files"), list):
            items = payload["files"]
        else:
            raise ValueError(
                "JSON 路径清单必须是数组，"
                "或包含 paths/files 数组的对象。"
            )

        return [
            normalize_path_item(item, paths_file.parent)
            for item in items
        ]

    except json.JSONDecodeError:
        raw_paths = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

        return [
            normalize_path(raw_path, paths_file.parent)
            for raw_path in raw_paths
        ]


def unique_paths(paths: Iterable[Path]) -> List[Path]:
    result: List[Path] = []
    seen = set()

    for path in paths:
        key = str(path.resolve(strict=False)).casefold()

        if key in seen:
            continue

        seen.add(key)
        result.append(path)

    return result


def collect_paths(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = []

    for raw_path in args.path or []:
        paths.append(normalize_path(raw_path, Path.cwd()))

    if args.paths_file:
        paths.extend(load_paths_file(Path(args.paths_file)))

    if args.input_dir:
        input_dir = Path(args.input_dir)
        pattern = "**/*.txt" if args.recursive else "*.txt"
        paths.extend(sorted(input_dir.glob(pattern)))

    paths = unique_paths(paths)

    if not paths:
        raise ValueError(
            "没有获得任何输入文件。"
            "请使用 --path、--paths-file 或 --input-dir。"
        )

    return paths


def clamp_confidence(value: Any) -> float:
    """
    将内部置信度转换为 0~1。
    当前置信度表示规则决策强度，不是经过校准的统计概率。
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0

    return round(max(0.0, min(1.0, number)), 4)



def get_candidate_confidence(
    match_result: Dict[str, Any],
    final_known_class: Optional[str],
    final_category: Optional[str],
) -> Optional[float]:
    """
    获取最终小类对应的原始候选置信度。

    优先规则：
    1. 在 known_class_candidates 中寻找与最终 known class 完全一致的候选；
    2. 若没有最终 known class，则不返回候选置信度；
    3. 不盲目使用全局最高候选，避免最高候选与最终小类不一致。
    """
    if not final_known_class:
        return None

    candidates = match_result.get("known_class_candidates") or []
    matched_candidates: List[Dict[str, Any]] = []

    for item in candidates:
        if not isinstance(item, dict):
            continue

        candidate_label = (
            item.get("label")
            or item.get("ship_class")
            or item.get("name")
        )
        candidate_category = item.get("category")

        if candidate_label != final_known_class:
            continue

        if final_category and candidate_category and candidate_category != final_category:
            continue

        matched_candidates.append(item)

    if not matched_candidates:
        return None

    best_item = max(
        matched_candidates,
        key=lambda item: float(item.get("confidence", 0.0) or 0.0),
    )

    try:
        return float(best_item.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None


def extract_minimal_result(
    file_path: Path,
    match_result: Dict[str, Any],
) -> Dict[str, Any]:
    final_decision = match_result.get("final_decision") or {}
    category_result = match_result.get("category_result") or {}
    known_class_result = match_result.get("known_class_result") or {}
    open_set_result = match_result.get("open_set_result") or {}

    category_label = (
        final_decision.get("primary_category")
        or category_result.get("label")
        or known_class_result.get("category")
    )

    category_confidence = clamp_confidence(
        category_result.get(
            "confidence",
            final_decision.get("confidence", 0.0),
        )
    )

    known_class_label = (
        final_decision.get("primary_class")
        or known_class_result.get("label")
        or known_class_result.get("ship_class")
    )

    is_open_set = bool(open_set_result.get("is_unknown", False))

    if final_decision.get("result_type") == "category_unknown":
        is_open_set = True

    if is_open_set or not known_class_label:
        small_class_label = (
            open_set_result.get("unknown_scope")
            or (
                f"{category_label}类别内未知类"
                if category_label
                else "类别内未知类"
            )
        )

        # 类别内未知不是 known_class_candidates 中的某个已知舰级，
        # 因此不能把“最高已知候选分”当成未知小类置信度。
        # 这里继续使用最终开放集决策强度。
        small_class_confidence = clamp_confidence(
            final_decision.get(
                "confidence",
                category_result.get("confidence", 0.0),
            )
        )
    else:
        small_class_label = known_class_label

        # 已知小类置信度优先使用“最终小类在 known_class_candidates
        # 中对应候选的原始置信度”，不再直接使用后处理同步后的
        # known_class_result.confidence。
        candidate_confidence = get_candidate_confidence(
            match_result=match_result,
            final_known_class=known_class_label,
            final_category=category_label,
        )

        if candidate_confidence is not None:
            small_class_confidence = clamp_confidence(candidate_confidence)
        else:
            # 极少数补丁路径可能没有保留候选项，此时才回退。
            small_class_confidence = clamp_confidence(
                known_class_result.get(
                    "confidence",
                    final_decision.get("confidence", 0.0),
                )
            )

    return {
        "file_path": str(file_path.resolve()),
        "status": "success",
        "category_result": category_label,
        "category_confidence": category_confidence,
        "small_class_result": small_class_label,
        "small_class_confidence": small_class_confidence,
    }


async def infer_one(
    file_path: Path,
    class_data_path: str,
) -> Dict[str, Any]:
    input_text, _encoding = read_text_file(file_path)

    if not input_text.strip():
        raise ValueError(f"文本文件为空：{file_path}")

    parsed_text = await direct_text_parse_v2(input_text)
    parsed_object = extract_json_object(parsed_text)
    observed_attributes = normalize_observed_attributes(parsed_object)

    # 保持与 batch_test_sim_text.py 相同：
    # 将原始文本交给 run_deepseek 的后处理逻辑。
    observed_attributes["_META"] = {
        "raw_text": input_text,
    }

    # 执行 schema 校验；若有普通字段问题，仍交给分类流程处理。
    validate_observed_schema(
        {
            key: value
            for key, value in observed_attributes.items()
            if key != "_META"
        }
    )

    match_result = hierarchical_class_match(
        class_data_path,
        observed_attributes,
    )

    return extract_minimal_result(
        file_path=file_path,
        match_result=match_result,
    )


async def main_async(args: argparse.Namespace) -> int:
    paths = collect_paths(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    print(f"共读取到 {len(paths)} 个文本文件。")
    print("推理模式：run_deepseek baseline（不使用 RAG）")

    for index, file_path in enumerate(paths, start=1):
        print(f"\n[{index}/{len(paths)}] 正在识别：{file_path}")
        started = time.perf_counter()

        try:
            result = await infer_one(
                file_path=file_path,
                class_data_path=args.class_data,
            )

            results.append(result)

            print(
                f"  大类={result['category_result']} "
                f"({result['category_confidence']:.4f}) | "
                f"小类={result['small_class_result']} "
                f"({result['small_class_confidence']:.4f}) | "
                f"耗时={time.perf_counter() - started:.1f}s"
            )

        except Exception as exc:
            results.append({
                "file_path": str(file_path.resolve(strict=False)),
                "status": "error",
                "category_result": None,
                "category_confidence": 0.0,
                "small_class_result": None,
                "small_class_confidence": 0.0,
                "error_message": str(exc),
            })

            print(
                f"  失败：{type(exc).__name__}: {exc} | "
                f"耗时={time.perf_counter() - started:.1f}s"
            )

            if args.fail_fast:
                break

        if args.sleep > 0 and index < len(paths):
            await asyncio.sleep(args.sleep)

    # 正式输出只保留 results 数组。
    output_payload = {
        "results": results,
    }

    output_path.write_text(
        json.dumps(
            output_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    success_count = sum(
        1 for item in results
        if item.get("status") == "success"
    )
    failed_count = len(results) - success_count

    print("\n处理完成。")
    print(f"成功：{success_count}，失败：{failed_count}")
    print(f"结果 JSON：{output_path.resolve()}")

    return 0 if failed_count == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "基于 run_deepseek.py 批量读取文本文件，"
            "输出大类、小类及置信度 JSON。"
        )
    )

    parser.add_argument(
        "--path",
        action="append",
        help="单个文本文件路径，可重复传入。",
    )

    parser.add_argument(
        "--paths-file",
        help="JSON 路径清单或每行一个路径的文本文件。",
    )

    parser.add_argument(
        "--input-dir",
        help="扫描目录下的 .txt 文件。",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="配合 --input-dir 递归扫描子目录。",
    )

    parser.add_argument(
        "--class-data",
        default="./class_data.txt",
    )

    parser.add_argument(
        "--output",
        default="./ship_inference_results.json",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="相邻文件的推理间隔秒数，默认 1 秒。",
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="遇到第一条错误后立即停止。",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
