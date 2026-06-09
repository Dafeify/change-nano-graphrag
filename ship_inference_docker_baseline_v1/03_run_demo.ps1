Set-Location $PSScriptRoot

python .\demo\demo_client.py `
  --paths-file .\demo\demo_paths.json `
  --api-url http://127.0.0.1:8000/infer `
  --output .\demo\ship_inference_results.json
