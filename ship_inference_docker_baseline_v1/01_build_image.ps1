Set-Location $PSScriptRoot

docker build `
  -t ship-text-inference:baseline-v1 `
  .\docker_app
