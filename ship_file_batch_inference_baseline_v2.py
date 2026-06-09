# -*- coding: utf-8 -*-
r"""
ship_file_batch_inference_baseline_v2.py

基于 run_deepseek.py 的无 RAG 批量推理入口。

支持的输入来源：
1. 本机绝对路径：D:\data\ship_001.txt
2. 相对路径：.\inputs\ship_001.txt
3. Windows 网络共享 UNC：\\server\share\ship_001.txt
4. file:// URI：file:///D:/data/ship_001.txt
5. HTTP/HTTPS URL：https://example.com/ship_001.txt
6. JSON 对象中的 path / url 字段

注意：
- “本地路径”不要求位于项目目录，只要运行本程序的进程有权限访问即可。
- 如果老师的工具与本程序不在同一台机器，老师电脑上的 D:\xxx 路径对本程序通常不可见；
  此时应使用共享目录、挂载目录、URL，或由上层工具先上传文件并传入临时路径。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse, unquote
from urllib.request import url2pathname

import httpx

from run_deepseek import (
    direct_text_parse_v2,
    extract_json_object,
    normalize_observed_attributes,
    validate_observed_schema,
    hierarchical_class_match,
)

SUPPORTED_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")


def decode_bytes_auto(data: bytes, source: str) -> Tuple[str, str]:
    last_error: Optional[Exception] = None
    for encoding in SUPPORTED_ENCODINGS:
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeError(
        f"无法识别文本编码：{source}。已尝试 {', '.join(SUPPORTED_ENCODINGS)}"
    ) from last_error


def read_local_text(path: Path) -> Tuple[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    if not path.is_file():
        raise ValueError(f"路径不是普通文件：{path}")
    return decode_bytes_auto(path.read_bytes(), str(path))


async def read_http_text(url: str) -> Tuple[str, str]:
    timeout = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        return decode_bytes_auto(response.content, url)


def file_uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme.lower() != "file":
        raise ValueError(f"不是 file URI：{uri}")

    # Windows file:///D:/x.txt 或 file://server/share/x.txt
    path_text = url2pathname(unquote(parsed.path))
    if parsed.netloc:
        return Path(f"//{parsed.netloc}{path_text}")
    if path_text.startswith("/") and len(path_text) >= 3 and path_text[2] == ":":
        path_text = path_text[1:]
    return Path(path_text)


def classify_source(raw: str, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    text = str(raw).strip()
    parsed = urlparse(text)
    scheme = parsed.scheme.lower()

    if scheme in {"http", "https"}:
        return {"kind": "url", "source": text, "display": text}

    if scheme == "file":
        path = file_uri_to_path(text)
        return {"kind": "path", "source": path, "display": text}

    path = Path(text)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path

    # Path.resolve(strict=False) 不要求文件已存在，UNC 也可保留。
    return {
        "kind": "path",
        "source": path,
        "display": str(path),
    }


def normalize_item(item: Any, base_dir: Optional[Path]) -> Dict[str, Any]:
    if isinstance(item, str):
        source = classify_source(item, base_dir)
        source["id"] = None
        return source

    if not isinstance(item, dict):
        raise ValueError(f"路径清单元素必须是字符串或对象，实际为：{type(item).__name__}")

    item_id = item.get("id")
    if item.get("path"):
        source = classify_source(str(item["path"]), base_dir)
    elif item.get("url"):
        source = classify_source(str(item["url"]), base_dir)
    elif item.get("source"):
        source = classify_source(str(item["source"]), base_dir)
    else:
        raise ValueError(f"路径对象缺少 path/url/source：{item}")

    source["id"] = item_id
    return source


def read_paths_file(path_file: Path) -> List[Dict[str, Any]]:
    text, _ = read_local_text(path_file)
    stripped = text.strip()
    if not stripped:
        return []

    try:
        payload = json.loads(stripped)
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict) and isinstance(payload.get("files"), list):
            items = payload["files"]
        elif isinstance(payload, dict) and isinstance(payload.get("paths"), list):
            items = payload["paths"]
        else:
            raise ValueError(
                "JSON 路径文件必须是数组，或包含 files/paths 数组的对象。"
            )
        return [normalize_item(item, path_file.parent) for item in items]
    except json.JSONDecodeError:
        items = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        return [normalize_item(item, path_file.parent) for item in items]


def unique_sources(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        source = item["source"]
        if item["kind"] == "path":
            key = f"path:{str(Path(source).resolve(strict=False)).casefold()}"
        else:
            key = f"url:{str(source)}"
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def collect_sources(args: argparse.Namespace) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    for raw in args.path or []:
        items.append(normalize_item({"path": raw}, Path.cwd()))

    for raw in args.url or []:
        items.append(normalize_item({"url": raw}, None))

    if args.paths_file:
        items.extend(read_paths_file(Path(args.paths_file)))

    if args.input_dir:
        input_dir = Path(args.input_dir)
        pattern = "**/*.txt" if args.recursive else "*.txt"
        for path in sorted(input_dir.glob(pattern)):
            items.append({
                "id": None,
                "kind": "path",
                "source": path,
                "display": str(path),
            })

    items = unique_sources(items)
    if not items:
        raise ValueError(
            "没有获得输入。请使用 --path、--url、--paths-file 或 --input-dir。"
        )
    return items


async def load_source_text(item: Dict[str, Any]) -> Tuple[str, str]:
    if item["kind"] == "url":
        return await read_http_text(str(item["source"]))
    return read_local_text(Path(item["source"]))


def clamp_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(1.0, number)), 4)


def extract_business_result(
    item: Dict[str, Any],
    text: str,
    encoding: str,
    match_result: Dict[str, Any],
    include_debug: bool,
    parsed_obj: Dict[str, Any],
    observed: Dict[str, Any],
    schema_errors: List[str],
) -> Dict[str, Any]:
    final_decision = match_result.get("final_decision") or {}
    category_result = match_result.get("category_result") or {}
    known_result = match_result.get("known_class_result") or {}
    open_set_result = match_result.get("open_set_result") or {}

    category = (
        final_decision.get("primary_category")
        or category_result.get("label")
        or known_result.get("category")
    )
    category_confidence = clamp_confidence(
        category_result.get("confidence", final_decision.get("confidence", 0.0))
    )

    known_class = (
        final_decision.get("primary_class")
        or known_result.get("label")
        or known_result.get("ship_class")
    )
    open_set = bool(open_set_result.get("is_unknown", False))
    if final_decision.get("result_type") == "category_unknown":
        open_set = True

    if open_set or not known_class:
        small_class = (
            open_set_result.get("unknown_scope")
            or (f"{category}类别内未知类" if category else "类别内未知类")
        )
        known_output = None
        small_confidence = clamp_confidence(
            final_decision.get("confidence", category_confidence)
        )
        result_type = "category_unknown"
    else:
        small_class = known_class
        known_output = known_class
        small_confidence = clamp_confidence(
            known_result.get("confidence", final_decision.get("confidence", 0.0))
        )
        result_type = "known_class"

    source_display = item["display"]
    source_name = (
        Path(item["source"]).name
        if item["kind"] == "path"
        else Path(urlparse(str(item["source"])).path).name
    )

    result: Dict[str, Any] = {
        "id": item.get("id"),
        "source_type": item["kind"],
        "source": source_display,
        "file_name": source_name,
        "status": "success",
        "result_type": result_type,
        "category_result": category,
        "category_confidence": category_confidence,
        "small_class_result": small_class,
        "small_class_confidence": small_confidence,
        "known_class_result": known_output,
        "open_set": open_set,
        "unknown_scope": open_set_result.get("unknown_scope") if open_set else None,
        "text_encoding": encoding,
        "text_length": len(text),
        "decision_message": final_decision.get("message"),
    }

    if include_debug:
        result["debug"] = {
            "input_text": text,
            "parsed_object": parsed_obj,
            "observed_attributes": observed,
            "schema_errors": schema_errors,
            "match_result": match_result,
        }
    return result


async def infer_one(
    item: Dict[str, Any],
    class_data_path: str,
    include_debug: bool,
) -> Dict[str, Any]:
    text, encoding = await load_source_text(item)
    if not text.strip():
        raise ValueError(f"输入文本为空：{item['display']}")

    parse_text = await direct_text_parse_v2(text)
    parsed_obj = extract_json_object(parse_text)
    observed = normalize_observed_attributes(parsed_obj)
    observed["_META"] = {"raw_text": text}

    schema_errors = validate_observed_schema(
        {key: value for key, value in observed.items() if key != "_META"}
    )
    match_result = hierarchical_class_match(class_data_path, observed)

    return extract_business_result(
        item=item,
        text=text,
        encoding=encoding,
        match_result=match_result,
        include_debug=include_debug,
        parsed_obj=parsed_obj,
        observed=observed,
        schema_errors=schema_errors,
    )


async def main_async(args: argparse.Namespace) -> int:
    items = collect_sources(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    succeeded = 0
    failed = 0

    print(f"共收到 {len(items)} 个输入来源。")
    print("推理引擎：run_deepseek baseline（不使用 RAG）")

    for index, item in enumerate(items, start=1):
        print(f"\n[{index}/{len(items)}] 正在处理：{item['display']}")
        started = time.perf_counter()
        try:
            result = await infer_one(
                item=item,
                class_data_path=args.class_data,
                include_debug=args.include_debug,
            )
            result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            results.append(result)
            succeeded += 1
            print(
                f"  大类={result['category_result']} "
                f"({result['category_confidence']:.4f}) | "
                f"小类={result['small_class_result']} "
                f"({result['small_class_confidence']:.4f})"
            )
        except Exception as exc:
            failed += 1
            results.append({
                "id": item.get("id"),
                "source_type": item["kind"],
                "source": item["display"],
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc() if args.include_debug else None,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            })
            print(f"  失败：{type(exc).__name__}: {exc}")
            if args.fail_fast:
                break

        if args.sleep > 0 and index < len(items):
            await asyncio.sleep(args.sleep)

    payload = {
        "schema_version": "1.1",
        "inference_engine": "run_deepseek_baseline",
        "uses_rag": False,
        "uses_training": False,
        "supported_sources": [
            "local_absolute_path",
            "local_relative_path",
            "windows_unc_path",
            "file_uri",
            "http_url",
            "https_url",
        ],
        "confidence_note": (
            "置信度为当前规则匹配系统输出的 0~1 决策强度，"
            "尚未经过概率校准。"
        ),
        "summary": {
            "requested": len(items),
            "processed": len(results),
            "succeeded": succeeded,
            "failed": failed,
        },
        "results": results,
    }

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n处理完成。")
    print(f"成功：{succeeded}，失败：{failed}")
    print(f"结果 JSON：{output_path.resolve()}")
    return 0 if failed == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="通过路径或 URL 批量识别舰船文本，并输出 JSON。"
    )
    parser.add_argument("--path", action="append", help="本地/共享文件路径，可重复。")
    parser.add_argument("--url", action="append", help="HTTP/HTTPS 文本 URL，可重复。")
    parser.add_argument("--paths-file", help="JSON 或逐行路径清单。")
    parser.add_argument("--input-dir", help="扫描目录下的 .txt 文件。")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--class-data", default="./class_data.txt")
    parser.add_argument("--output", default="./ship_inference_results.json")
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--include-debug", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
