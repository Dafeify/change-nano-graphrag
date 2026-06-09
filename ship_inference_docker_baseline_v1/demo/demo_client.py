# -*- coding: utf-8 -*-
"""
宿主机侧 demo：
输入一个或多个文本文件路径，读取内容后调用 Docker API，
把结果保存为 JSON 并打印。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SUPPORTED_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")


def read_text_auto(path: Path) -> Tuple[str, str]:
    data = path.read_bytes()
    last_error: Optional[Exception] = None

    for encoding in SUPPORTED_ENCODINGS:
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeError(
        f"无法识别文本编码：{path}。已尝试 {', '.join(SUPPORTED_ENCODINGS)}"
    ) from last_error


def normalize_path(raw: str, base_dir: Optional[Path] = None) -> Path:
    path = Path(str(raw).strip())
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


def load_paths_file(paths_file: Path) -> List[Path]:
    text, _ = read_text_auto(paths_file)
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
                "JSON 路径清单必须是数组，或包含 paths/files 数组的对象。"
            )

        output: List[Path] = []
        for item in items:
            if isinstance(item, str):
                output.append(normalize_path(item, paths_file.parent))
            elif isinstance(item, dict):
                raw_path = (
                    item.get("path")
                    or item.get("file_path")
                    or item.get("source")
                )
                if not raw_path:
                    raise ValueError(f"路径对象缺少 path/file_path/source：{item}")
                output.append(normalize_path(raw_path, paths_file.parent))
            else:
                raise ValueError(f"不支持的路径项：{item}")
        return output

    except json.JSONDecodeError:
        return [
            normalize_path(line.strip(), paths_file.parent)
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
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

    for raw in args.path or []:
        paths.append(normalize_path(raw, Path.cwd()))

    if args.paths_file:
        paths.extend(load_paths_file(Path(args.paths_file)))

    paths = unique_paths(paths)
    if not paths:
        raise ValueError("请至少使用一次 --path，或指定 --paths-file。")
    return paths


def build_payload(paths: List[Path]) -> Dict[str, Any]:
    files: List[Dict[str, str]] = []

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"文件不存在：{path}")
        if not path.is_file():
            raise ValueError(f"不是普通文件：{path}")

        text, _ = read_text_auto(path)
        if not text.strip():
            raise ValueError(f"文本文件为空：{path}")

        files.append({
            "file_path": str(path.resolve()),
            "content": text,
        })

    return {"files": files}


def post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"无法连接 Docker 推理服务：{url}。请确认容器已启动。"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="输入一个或多个文本文件路径，调用 Docker 推理服务。"
    )
    parser.add_argument("--path", action="append", help="文本文件路径，可重复。")
    parser.add_argument("--paths-file", help="JSON 清单或每行一个路径。")
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000/infer",
    )
    parser.add_argument(
        "--output",
        default="./ship_inference_results.json",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        paths = collect_paths(args)
        payload = build_payload(paths)

        print(f"准备提交 {len(paths)} 个文本文件到：{args.api_url}")
        result = post_json(args.api_url, payload, args.timeout)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\nDocker 推理结果：")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\nJSON 文件已保存：{output_path.resolve()}")
        return 0

    except Exception as exc:
        print(f"执行失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
