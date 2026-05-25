# 1. 彻底修复 Windows 控制台编码问题
import sys
import os

# 强制 Python 进入 UTF-8 模式
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LANG"] = "en_US.UTF-8"
os.environ["LC_ALL"] = "en_US.UTF-8"

# 2. 修补 httpx 的 header 编码函数，使其支持 UTF-8
import httpx._models
_original_normalize = httpx._models._normalize_header_value
def _utf8_normalize(value, encoding=None):
    if isinstance(value, bytes):
        return value
    # 直接使用 UTF-8 编码，而不是 ASCII
    return value.encode("utf-8")
httpx._models._normalize_header_value = _utf8_normalize

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import asyncio

# Windows 下使用更稳定的 Selector 事件循环
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# ==================== 本地 Embedding 模型 ====================
from sentence_transformers import SentenceTransformer
import numpy as np
from nano_graphrag._utils import wrap_embedding_func_with_attrs

# 加载中文优化的 Embedding 模型（首次运行会自动下载约 1.3GB）
EMBED_MODEL = SentenceTransformer("BAAI/bge-large-zh-v1.5")

# 获取模型的向量维度和最大 token 长度
_embedding_dim = EMBED_MODEL.get_sentence_embedding_dimension()   # 1024
_max_token_size = EMBED_MODEL.max_seq_length                     # 512

@wrap_embedding_func_with_attrs(embedding_dim=_embedding_dim, max_token_size=_max_token_size)
async def local_embedding(texts: list[str]) -> np.ndarray:
    """本地 Embedding 函数，使用 BAAI/bge-large-zh-v1.5 模型"""
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)


from openai import AsyncOpenAI
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash

# --- 模型配置 ---
MODEL = "deepseek-ai/DeepSeek-V3"
BASE_URL = "https://api.siliconflow.cn/v1"
API_KEY = ""  # 临时硬编码，测试完记得改回环境变量

# --- 自定义大模型调用函数 ---
async def siliconflow_llm_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # 处理缓存
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        **kwargs
    )
    result = response.choices[0].message.content

    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": MODEL}})

    return result

# --- 主程序 ---
async def main():
    graph_func = GraphRAG(
        working_dir="./ship_index",
        best_model_func=siliconflow_llm_complete,
        cheap_model_func=siliconflow_llm_complete,
        best_model_id=MODEL,
        cheap_model_id=MODEL,
        embedding_func=local_embedding,   # 启用本地 Embedding
    )

    with open("./naval_data.txt", "r", encoding='utf-8') as f:
        await graph_func.ainsert(f.read())

    print("=" * 50)
    print("全局查询结果：")
    print(await graph_func.aquery("这则文本的核心主题是什么？"))
    print("=" * 50)
    print("局部查询结果：")
    print(await graph_func.aquery("CVN-72 的雷达是什么型号的？", param=QueryParam(mode="local")))
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())
