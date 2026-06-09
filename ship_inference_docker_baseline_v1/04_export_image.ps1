Set-Location $PSScriptRoot

docker save `
  -o .\ship-text-inference_baseline-v1.tar `
  ship-text-inference:baseline-v1

Write-Host "镜像已导出：$PSScriptRoot\ship-text-inference_baseline-v1.tar"
