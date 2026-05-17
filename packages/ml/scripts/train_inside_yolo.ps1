param(
  [string] $Weights = "F:\lenta_price_vision\models\best.pt",
  [string] $Data = "F:\lenta_price_vision\packages\ml\src\ml\runs\datasets\inside_price_tag_yolo\data.yaml",
  [string] $RunName = "inside_price_tag_yolo_best",
  [int] $Epochs = 200,
  [int] $ImageSize = 960,
  [int] $Batch = 4,
  [string] $Device = "0"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Weights)) {
  throw "Weights not found: $Weights"
}

if (-not (Test-Path -LiteralPath $Data)) {
  throw "Dataset data.yaml not found: $Data"
}

Push-Location "F:\lenta_price_vision\packages\ml"
try {
  & uv run yolo detect train `
    "model=$Weights" `
    "data=$Data" `
    "epochs=$Epochs" `
    "imgsz=$ImageSize" `
    "batch=$Batch" `
    "device=$Device" `
    "project=F:\lenta_price_vision\packages\ml\runs\inside" `
    "name=$RunName" `
    "patience=40" `
    "cos_lr=True" `
    "close_mosaic=20" `
    "hsv_h=0.010" `
    "hsv_s=0.35" `
    "hsv_v=0.25" `
    "degrees=3.0" `
    "translate=0.05" `
    "scale=0.25" `
    "shear=1.0" `
    "perspective=0.0002" `
    "fliplr=0.0" `
    "mosaic=0.25" `
    "mixup=0.0" `
    "copy_paste=0.0" `
    "workers=2" `
    "seed=42" `
    "cache=True" `
    "plots=True"
}
finally {
  Pop-Location
}
