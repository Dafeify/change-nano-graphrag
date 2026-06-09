# -*- coding: utf-8 -*-
"""
app_server.py

Docker 容器内的无 RAG 舰船文本推理 HTTP 服务。

接口：
- GET  /health
- POST /infer

请求示例：
{
  "files": [
    {
      "file_path": "D:\\ship_texts\\001.txt",
      "content": "舰船文本内容"
    }
  ]
}

注意：
- file_path 只用于结果关联；
- 容器不读取宿主机路径；
- demo 在宿主机读取文本内容后发送给容器。
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from run_deepseek import (
    direct_text_parse_v2,
    extract_json_object,
    normalize_observed_attributes,
    validate_observed_schema,
    hierarchical_class_match,
)


APP_VERSION = "1.0.0"
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "50000"))
INFERENCE_LOCK = asyncio.Lock()


class InputFile(BaseModel):
    file_path: str = Field(..., description="原始文本文件路径，仅用于结果关联")
    content: str = Field(..., description="文本文件内容")


class InferRequest(BaseModel):
    files: List[InputFile]


def clamp_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(1.0, number)), 4)


def get_final_candidate_confidence(
    match_result: Dict[str, Any],
    final_class: Optional[str],
    final_category: Optional[str],
) -> Optional[float]:
    if not final_class:
        return None

    matched: List[Dict[str, Any]] = []
    for item in match_result.get("known_class_candidates") or []:
        if not isinstance(item, dict):
            continue

        label = item.get("label") or item.get("ship_class") or item.get("name")
        category = item.get("category")

        if label != final_class:
            continue
        if final_category and category and category != final_category:
            continue
        matched.append(item)

    if not matched:
        return None

    best = max(
        matched,
        key=lambda item: float(item.get("confidence", 0.0) or 0.0),
    )
    try:
        return float(best.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None


def extract_result(file_path: str, match_result: Dict[str, Any]) -> Dict[str, Any]:
    final_decision = match_result.get("final_decision") or {}
    category_result = match_result.get("category_result") or {}
    known_result = match_result.get("known_class_result") or {}
    open_set_result = match_result.get("open_set_result") or {}

    category_label = (
        final_decision.get("primary_category")
        or category_result.get("label")
        or known_result.get("category")
    )
    category_confidence = clamp_confidence(
        category_result.get("confidence", final_decision.get("confidence", 0.0))
    )

    known_label = (
        final_decision.get("primary_class")
        or known_result.get("label")
        or known_result.get("ship_class")
    )

    is_open_set = bool(open_set_result.get("is_unknown", False))
    if final_decision.get("result_type") == "category_unknown":
        is_open_set = True

    if is_open_set or not known_label:
        small_class_label = (
            open_set_result.get("unknown_scope")
            or (
                f"{category_label}类别内未知类"
                if category_label
                else "类别内未知类"
            )
        )
        small_class_confidence = clamp_confidence(
            final_decision.get("confidence", category_result.get("confidence", 0.0))
        )
    else:
        small_class_label = known_label
        candidate_confidence = get_final_candidate_confidence(
            match_result,
            final_class=known_label,
            final_category=category_label,
        )
        if candidate_confidence is not None:
            small_class_confidence = clamp_confidence(candidate_confidence)
        else:
            small_class_confidence = clamp_confidence(
                known_result.get(
                    "confidence",
                    final_decision.get("confidence", 0.0),
                )
            )

    return {
        "file_path": file_path,
        "status": "success",
        "category_result": category_label,
        "category_confidence": category_confidence,
        "small_class_result": small_class_label,
        "small_class_confidence": small_class_confidence,
    }


async def infer_text(file_path: str, content: str) -> Dict[str, Any]:
    text = str(content or "").strip()
    if not text:
        raise ValueError("文本内容为空")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"文本长度超过限制：{len(text)} > {MAX_TEXT_CHARS}")

    async with INFERENCE_LOCK:
        parsed_text = await direct_text_parse_v2(text)
        parsed_object = extract_json_object(parsed_text)
        observed = normalize_observed_attributes(parsed_object)
        observed["_META"] = {"raw_text": text}

        validate_observed_schema(
            {
                key: value
                for key, value in observed.items()
                if key != "_META"
            }
        )

        match_result = hierarchical_class_match("./class_data.txt", observed)

    return extract_result(file_path, match_result)


app = FastAPI(
    title="Ship Text Inference API",
    version=APP_VERSION,
    description="基于 run_deepseek.py 的无 RAG 舰船大类/小类识别服务",
)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": APP_VERSION,
        "uses_rag": False,
        "uses_training": False,
    }


@app.post("/infer")
async def infer(request: InferRequest) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []

    for item in request.files:
        try:
            result = await infer_text(item.file_path, item.content)
        except Exception as exc:
            result = {
                "file_path": item.file_path,
                "status": "error",
                "category_result": None,
                "category_confidence": 0.0,
                "small_class_result": None,
                "small_class_confidence": 0.0,
                "error_message": str(exc),
            }
        results.append(result)

    return {"results": results}
