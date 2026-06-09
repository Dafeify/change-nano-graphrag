# 舰船文本推理 Docker 集成包（Baseline / 无 RAG）

## 架构

- `docker_app/`：容器内推理服务。
- `demo/demo_client.py`：容器外 Python demo。

demo 接收本地文本文件路径，先在宿主机读取文件内容，再调用 Docker API。
因此 Docker 容器不需要直接访问宿主机上的 `D:\xxx` 路径。

## 当前封装内容

- `run_deepseek.py`
- `schema_config.py`
- `class_data.txt`
- FastAPI HTTP 服务
- Python 运行环境

当前 DeepSeek 通过 SiliconFlow 远程 API 调用，因此容器仍需：

- 能访问互联网；
- 启动时传入 `SILICONFLOW_API_KEY`。

本包：

- 不使用 RAG；
- 不涉及训练；
- 不包含 DeepSeek 模型权重。

## 运行步骤

### 1. 创建环境变量文件

```powershell
Set-Location .\docker_app
Copy-Item .\.env.docker.example .\.env.docker
```

编辑 `.env.docker`，填入真实 Key。

### 2. 构建镜像

回到包根目录：

```powershell
.\01_build_image.ps1
```

### 3. 启动容器

```powershell
.\02_run_container.ps1
```

检查：

```powershell
docker ps
Invoke-RestMethod http://127.0.0.1:8000/health
```

### 4. 运行 demo

```powershell
.\03_run_demo.ps1
```

demo 会保存并打印：

```text
demo\ship_inference_results.json
```

### 5. 正式传多个路径

```powershell
python .\demo\demo_client.py `
  --path "D:\teacher_ship_texts\001.txt" `
  --path "D:\teacher_ship_texts\002.txt" `
  --api-url http://127.0.0.1:8000/infer `
  --output .\teacher_ship_results.json
```

## 输出格式

```json
{
  "results": [
    {
      "file_path": "D:\\teacher_ship_texts\\001.txt",
      "status": "success",
      "category_result": "驱逐舰",
      "category_confidence": 0.926,
      "small_class_result": "阿利·伯克级驱逐舰",
      "small_class_confidence": 0.9599
    }
  ]
}
```

## 导出并交付 Docker 镜像

```powershell
.\04_export_image.ps1
```

会生成：

```text
ship-text-inference_baseline-v1.tar
```

交付老师：

1. `ship-text-inference_baseline-v1.tar`
2. `demo` 目录
3. `.env.docker.example`

老师加载镜像：

```powershell
docker load -i .\ship-text-inference_baseline-v1.tar
```

创建自己的 `.env.docker` 后启动：

```powershell
docker run -d `
  --name ship-text-inference `
  -p 8000:8000 `
  --env-file .\.env.docker `
  ship-text-inference:baseline-v1
```
