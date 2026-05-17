param(
  [ValidateSet("train", "val", "both")]
  [string] $Split = "both"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$MlRoot = Join-Path $ProjectRoot "packages\ml"
$Dataset = Join-Path $MlRoot "src\ml\runs\datasets\lenta_yolo_49_43_26_prop8\data.yaml"

if (-not (Test-Path -LiteralPath $Dataset)) {
  throw "Dataset not found: $Dataset"
}

Push-Location $MlRoot
try {
  if ($Split -eq "train" -or $Split -eq "both") {
    & uv run ml-review-annotations --dataset $Dataset --split train
  }

  if ($Split -eq "val" -or $Split -eq "both") {
    & uv run ml-review-annotations --dataset $Dataset --split val
  }
}
finally {
  Pop-Location
}
