param(
  [int] $Epochs = 150,
  [int] $ImageSize = 1280,
  [int] $Batch = 4,
  [string] $Device = "0",
  [string] $RunName = "price_tag_yolo_49_43_26_best"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$MlRoot = Join-Path $ProjectRoot "packages\ml"
$Weights = Join-Path $ProjectRoot "models\best.pt"
$Dataset = Join-Path $MlRoot "src\ml\runs\datasets\lenta_yolo_49_43_26_prop8\data.yaml"
$Project = Join-Path $MlRoot "runs\lenta"

if (-not (Test-Path -LiteralPath $Weights)) {
  throw "Weights not found: $Weights"
}

if (-not (Test-Path -LiteralPath $Dataset)) {
  throw "Dataset not found: $Dataset"
}

Push-Location $MlRoot
try {
  & uv run yolo detect train `
    "model=$Weights" `
    "data=$Dataset" `
    "epochs=$Epochs" `
    "imgsz=$ImageSize" `
    "batch=$Batch" `
    "device=$Device" `
    "project=$Project" `
    "name=$RunName" `
    "patience=30" `
    "cos_lr=True" `
    "close_mosaic=15" `
    "hsv_h=0.015" `
    "hsv_s=0.50" `
    "hsv_v=0.30" `
    "degrees=8.0" `
    "translate=0.08" `
    "scale=0.45" `
    "shear=2.0" `
    "perspective=0.0008" `
    "fliplr=0.0" `
    "mosaic=0.55" `
    "mixup=0.05" `
    "copy_paste=0.0" `
    "workers=2" `
    "seed=42"
}
finally {
  Pop-Location
}
