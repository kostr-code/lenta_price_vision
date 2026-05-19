"""
scripts/dedupe_predictions.py — IoU-based NMS дедупликация YOLO label-файлов.

Используется после псевдолейблинга inside-моделью для очистки дублирующихся
bbox перед импортом в CVAT. Работает с обычными YOLO label-файлами (без score).

Запуск:
    uv run python scripts/dedupe_predictions.py \\
        --source runs/inside_pseudo_raw \\
        --out-dir runs/inside_pseudo_dedup \\
        --iou-threshold 0.75
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def nms(labels: list[str], iou_thr: float, class_aware: bool) -> list[str]:
    """Применить Non-Maximum Suppression к списку YOLO-строк.

    В label-файлах нет score — считаем все боксы равноценными (conf=1.0) и
    оставляем первый встретившийся при одинаковом классе (keep-first).
    При class_aware=True подавляем только боксы одного класса.
    """
    boxes = []
    for line in labels:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])
        conf = 1.0  # в label-файлах нет уверенности — считаем все равными
        boxes.append(
            (cls, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, conf, line)
        )

    # сортировка по классу, затем по conf убыванию (при равных — сохраняем порядок)
    boxes.sort(key=lambda b: (b[0], -b[5]))
    kept = []
    suppressed = set()

    for i, (cls_i, x1i, y1i, x2i, y2i, _, line_i) in enumerate(boxes):
        if i in suppressed:
            continue
        kept.append(line_i)
        for j in range(i + 1, len(boxes)):
            if j in suppressed:
                continue
            cls_j, x1j, y1j, x2j, y2j, _, _ = boxes[j]
            if class_aware and cls_i != cls_j:
                continue
            # вычисляем IoU через площадь пересечения и объединения
            ix1, iy1 = max(x1i, x1j), max(y1i, y1j)
            ix2, iy2 = min(x2i, x2j), min(y2i, y2j)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            ai = (x2i - x1i) * (y2i - y1i)
            aj = (x2j - x1j) * (y2j - y1j)
            union = ai + aj - inter
            if union > 0 and inter / union > iou_thr:
                suppressed.add(j)
    return kept


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source",
        required=True,
        help="Папка с YOLO label .txt файлами (подпапки поддерживаются)",
    )
    p.add_argument(
        "--out-dir", required=True, help="Куда писать дедуплицированные метки"
    )
    p.add_argument(
        "--iou-threshold",
        type=float,
        default=0.75,
        help="Порог IoU для подавления дублей",
    )
    p.add_argument(
        "--class-aware", action="store_true", help="NMS только внутри одного класса"
    )
    p.add_argument(
        "--clear-output", action="store_true", help="Очистить out-dir перед записью"
    )
    args = p.parse_args()

    src = Path(args.source)
    out = Path(args.out_dir)
    if args.clear_output and out.exists():
        shutil.rmtree(out)

    total_in, total_out = 0, 0
    for lbl_path in sorted(src.rglob("*.txt")):
        rel = lbl_path.relative_to(src)
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        lines = lbl_path.read_text(encoding="utf-8").splitlines()
        kept = nms(lines, args.iou_threshold, args.class_aware)
        dest.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        total_in += len(lines)
        total_out += len(kept)

    print(f"[dedup] {total_in} -> {total_out} labels ({total_in - total_out} removed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
