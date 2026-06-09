Set-Location $PSScriptRoot

$envFile = Join-Path $PSScriptRoot "docker_app\.env.docker"

if (-not (Test-Path $envFile)) {
    Write-Error "找不到 docker_app\.env.docker。请复制 .env.docker.example 并填写 SILICONFLOW_API_KEY。"
    exit 1
}

docker rm -f ship-text-inference 2>$null

docker run -d `
  --name ship-text-inference `
  -p 8000:8000 `
  --env-file $envFile `
  ship-text-inference:baseline-v1
