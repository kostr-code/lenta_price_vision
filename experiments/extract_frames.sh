#!/usr/bin/env bash
# Извлекает ВСЕ кадры из labeled видео как lossless PNG.
# Без масштабирования, без интерполяции, без фильтров.
# -noautorotate: кадры как хранятся в файле (без авто-поворота по метаданным).
set -e
cd "$(dirname "$0")"

declare -A VIDEOS=(
  ["frames_43_15"]="Данные/43_15/43_15.mp4"
  ["frames_25_12-20"]="Данные/25_12-20/25_12-20.mp4"
)

for outdir in "${!VIDEOS[@]}"; do
  video="${VIDEOS[$outdir]}"
  if [[ ! -f "$video" ]]; then
    echo "SKIP: $video не найден"
    continue
  fi

  frames=$(ffprobe -v error -select_streams v:0 \
    -show_entries stream=nb_frames \
    -of default=nokey=1:noprint_wrappers=1 "$video")
  fps=$(ffprobe -v error -select_streams v:0 \
    -show_entries stream=r_frame_rate \
    -of default=nokey=1:noprint_wrappers=1 "$video")

  echo ""
  echo "=== $video ==="
  echo "    Кадров: $frames  |  FPS: $fps"
  echo "    Вывод:  $outdir/"

  mkdir -p "$outdir"
  ffmpeg -noautorotate -i "$video" -vsync 0 "${outdir}/%06d.png"

  echo "    Готово: $(ls "$outdir"/*.png 2>/dev/null | wc -l) файлов"
done

echo ""
echo "Всё готово."
