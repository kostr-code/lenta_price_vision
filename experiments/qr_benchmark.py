"""
qr_benchmark.py — проверяем насколько хорошо читаются QR на реальных ценниках.

Использование:
    python qr_benchmark.py --csv 25_12-20.csv --video "данные/25_12-20/2.mp4"
    python qr_benchmark.py --csv 43_15.csv    --video "данные/43_15/43_15.mp4"

Зависимости:
    pip install opencv-python pyzbar zxing-cpp pandas numpy pillow
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ── попробуем оба декодера ──────────────────────────────────────────────────
try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    HAS_PYZBAR = True
except ImportError:
    HAS_PYZBAR = False
    print("[warn] pyzbar не найден, установи: pip install pyzbar")

try:
    import zxingcpp
    HAS_ZXING = True
except ImportError:
    HAS_ZXING = False
    print("[warn] zxing-cpp не найден, установи: pip install zxing-cpp")


# ── парсинг CSV Ленты ───────────────────────────────────────────────────────

def parse_lenta_csv(csv_path: str) -> pd.DataFrame:
    """Читает CSV разметки, фиксирует запятые в числах и пустые поля."""
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = df.columns.str.strip()

    # координаты — запятая как десятичный разделитель
    for col in ["x_min", "y_min", "x_max", "y_max"]:
        df[col] = df[col].str.replace(",", ".").astype(float)

    df["frame_timestamp"] = pd.to_numeric(df["frame_timestamp"], errors="coerce")
    return df


# ── извлечение кадра из видео ───────────────────────────────────────────────

def extract_frame(video_path: str, timestamp_ms: float) -> np.ndarray | None:
    """Возвращает кадр по временной метке (мс). Видео повёрнуто на 90°."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[error] Не могу открыть видео: {video_path}")
        return None

    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        return None

    # видео снято боком — поворачиваем CCW (камера была повёрнута CW)
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


# ── кроп ценника по bbox ────────────────────────────────────────────────────

def crop_tag(frame: np.ndarray, row: pd.Series, padding: int = 8) -> np.ndarray:
    """
    Вырезает ценник из кадра с небольшим отступом.
    CSV координаты — в пространстве оригинального (до поворота) кадра 3840×2160.
    После cv2.ROTATE_90_CLOCKWISE кадр становится 2160×3840:
        new_x = H_orig - 1 - y_orig  (H_orig = 2160)
        new_y = x_orig
    """
    h, w = frame.shape[:2]        # h=3840, w=2160 после CCW поворота
    W_ORIG = 3840

    # bbox в ориг. координатах
    ox1, oy1 = float(row.x_min), float(row.y_min)
    ox2, oy2 = float(row.x_max), float(row.y_max)

    # трансформируем в координаты повёрнутого (CCW) кадра:
    #   new_x = y_orig,  new_y = W_orig - 1 - x_orig
    nx1 = oy1
    nx2 = oy2
    ny1 = W_ORIG - 1 - ox2
    ny2 = W_ORIG - 1 - ox1

    x1 = max(0, int(nx1) - padding)
    y1 = max(0, int(ny1) - padding)
    x2 = min(w, int(nx2) + padding)
    y2 = min(h, int(ny2) + padding)
    return frame[y1:y2, x1:x2]


# ── preprocessing перед декодированием ─────────────────────────────────────

def preprocess_for_qr(crop: np.ndarray) -> list[np.ndarray]:
    """
    Возвращает несколько вариантов preprocessing — декодер попробует каждый.
    Чем больше вариантов, тем выше шанс прочитать сложный QR.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    variants = [gray]

    # контраст
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants.append(clahe.apply(gray))

    # sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    variants.append(sharpened)

    # бинаризация Otsu
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(bw)

    # апскейл x2 (помогает на маленьких QR)
    up = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    variants.append(up)

    return variants


# ── декодирование QR ────────────────────────────────────────────────────────

def decode_qr(crop: np.ndarray) -> dict:
    """
    Пробует оба декодера на всех preprocessing вариантах.
    Возвращает {'raw': str, 'fields': dict, 'decoder': str} или {}.
    """
    variants = preprocess_for_qr(crop)

    for img in variants:
        # pyzbar
        if HAS_PYZBAR:
            results = pyzbar_decode(img)
            for r in results:
                if r.type in ("QRCODE",):
                    parsed = parse_qr_content(r.data.decode("utf-8", errors="ignore"))
                    if parsed:
                        return {"raw": r.data.decode(), "fields": parsed, "decoder": "pyzbar"}

        # zxing-cpp
        if HAS_ZXING:
            results = zxingcpp.read_barcodes(img)
            for r in results:
                if "QR" in str(r.format):
                    parsed = parse_qr_content(r.text)
                    if parsed:
                        return {"raw": r.text, "fields": parsed, "decoder": "zxing"}

    return {}


def decode_barcode(crop: np.ndarray) -> str:
    """Пробует прочитать линейный штрихкод (EAN-13)."""
    variants = preprocess_for_qr(crop)
    for img in variants:
        if HAS_PYZBAR:
            for r in pyzbar_decode(img):
                if r.type in ("EAN13", "EAN8", "CODE128", "CODE39"):
                    return r.data.decode()
        if HAS_ZXING:
            for r in zxingcpp.read_barcodes(img):
                if "QR" not in str(r.format):
                    return r.text
    return ""


# ── парсинг содержимого QR Ленты ────────────────────────────────────────────

# Ленточный QR содержит поля в формате key=value&key=value
# или коротких алиасах: b=barcode, p1=price1, aP=actionPrice и т.д.

QR_FIELD_MAP = {
    # длинные имена           короткие алиасы
    "barcode":                "b",
    "price1":                 "p1",
    "price2":                 "p2",
    "price3":                 "p3",
    "price4":                 "p4",
    "wholesaleLevel1Count":   "wL1C",
    "wholesaleLevel1Price":   "wL1P",
    "wholesaleLevel2Count":   "wL2C",
    "wholesaleLevel2Price":   "wL2P",
    "actionPrice":            "aP",
    "actionCode":             "aC",
}

# маппинг из QR-ключа в CSV-колонку
QR_TO_CSV = {
    "barcode":               "qr_code_barcode",
    "b":                     "qr_code_barcode",
    "price1":                "price1_qr",
    "p1":                    "price1_qr",
    "price2":                "price2_qr",
    "p2":                    "price2_qr",
    "price3":                "price3_qr",
    "p3":                    "price3_qr",
    "price4":                "price4_qr",
    "p4":                    "price4_qr",
    "wholesaleLevel1Count":  "wholesale_level_1_count",
    "wL1C":                  "wholesale_level_1_count",
    "wholesaleLevel1Price":  "wholesale_level_1_price",
    "wL1P":                  "wholesale_level_1_price",
    "wholesaleLevel2Count":  "wholesale_level_2_count",
    "wL2C":                  "wholesale_level_2_count",
    "wholesaleLevel2Price":  "wholesale_level_2_price",
    "wL2P":                  "wholesale_level_2_price",
    "actionPrice":           "action_price_qr",
    "aP":                    "action_price_qr",
    "actionCode":            "action_code_qr",
    "aC":                    "action_code_qr",
}


def parse_qr_content(raw: str) -> dict:
    """
    Парсит строку QR-кода в словарь CSV-полей.
    Поддерживает форматы: key=val&key=val, JSON, и просто числа (штрихкод).
    """
    if not raw:
        return {}

    fields = {}

    # попытка 1: URL-query формат key=val&key=val
    if "=" in raw:
        for part in raw.replace(";", "&").split("&"):
            if "=" in part:
                k, _, v = part.partition("=")
                k, v = k.strip(), v.strip()
                csv_key = QR_TO_CSV.get(k)
                if csv_key:
                    fields[csv_key] = v

    # попытка 2: JSON
    if not fields and raw.startswith("{"):
        try:
            obj = json.loads(raw)
            for k, v in obj.items():
                csv_key = QR_TO_CSV.get(k)
                if csv_key:
                    fields[csv_key] = str(v)
        except json.JSONDecodeError:
            pass

    # попытка 3: просто штрихкод (13 цифр)
    if not fields and raw.strip().isdigit() and len(raw.strip()) == 13:
        fields["qr_code_barcode"] = raw.strip()

    return fields


# ── сравнение результата с ground truth ─────────────────────────────────────

QR_CSV_FIELDS = [
    "qr_code_barcode", "price1_qr", "price2_qr", "price3_qr", "price4_qr",
    "wholesale_level_1_count", "wholesale_level_1_price",
    "wholesale_level_2_count", "wholesale_level_2_price",
    "action_price_qr", "action_code_qr",
]


def compare_fields(predicted: dict, ground_truth: pd.Series) -> dict:
    """
    Сравнивает предсказанные поля с GT.
    Возвращает статистику совпадений.
    """
    results = {}
    for field in QR_CSV_FIELDS:
        gt_val = str(ground_truth.get(field, "")).strip()
        pr_val = str(predicted.get(field, "")).strip()

        # нормализуем числа: "252.63" == "252,63"
        gt_norm = gt_val.replace(",", ".").rstrip("0").rstrip(".")
        pr_norm = pr_val.replace(",", ".").rstrip("0").rstrip(".")

        if gt_val in ("нет", "nan", ""):
            results[field] = "skip"   # поля нет на ценнике — не считаем
        elif gt_norm and gt_norm == pr_norm:
            results[field] = "ok"
        elif pr_val == "":
            results[field] = "miss"   # не смогли прочитать
        else:
            results[field] = "wrong"  # прочитали, но неверно
    return results


# ── основной цикл ────────────────────────────────────────────────────────────

def run_benchmark(csv_path: str, video_path: str, save_crops: bool = False):
    print(f"\n{'='*60}")
    print(f"CSV:   {csv_path}")
    print(f"Video: {video_path}")
    print(f"{'='*60}\n")

    df = parse_lenta_csv(csv_path)
    print(f"Ценников в разметке: {len(df)}\n")

    crops_dir = Path("debug_crops")
    if save_crops:
        crops_dir.mkdir(exist_ok=True)

    # кэш кадров — не перечитываем видео для одного timestamp
    frame_cache: dict[float, np.ndarray] = {}

    stats = {
        "total": 0,
        "qr_decoded": 0,
        "barcode_decoded": 0,
        "field_ok": 0,
        "field_miss": 0,
        "field_wrong": 0,
        "field_skip": 0,
    }
    per_tag_results = []

    for idx, row in df.iterrows():
        ts = row["frame_timestamp"]
        tag_id = f"tag_{idx:03d}"

        # ── извлечь кадр ──────────────────────────────────────
        if ts not in frame_cache:
            frame = extract_frame(video_path, ts)
            if frame is None:
                print(f"[{tag_id}] ❌ кадр {ts}ms не найден")
                continue
            frame_cache[ts] = frame
        frame = frame_cache[ts]

        # ── кроп ──────────────────────────────────────────────
        crop = crop_tag(frame, row)
        if crop.size == 0:
            print(f"[{tag_id}] ❌ пустой кроп")
            continue

        stats["total"] += 1

        if save_crops:
            cv2.imwrite(str(crops_dir / f"{tag_id}.jpg"), crop)

        # ── QR ────────────────────────────────────────────────
        qr_result = decode_qr(crop)
        qr_fields = qr_result.get("fields", {})

        if qr_fields:
            stats["qr_decoded"] += 1
            decoder = qr_result.get("decoder", "?")
        else:
            decoder = "—"

        # ── штрихкод (линейный) ───────────────────────────────
        barcode_val = decode_barcode(crop)
        if barcode_val:
            stats["barcode_decoded"] += 1

        # ── сравнение с GT ────────────────────────────────────
        field_results = compare_fields(qr_fields, row)
        for status in field_results.values():
            if status != "skip":
                stats[f"field_{status}"] += 1

        ok_count    = sum(1 for s in field_results.values() if s == "ok")
        total_count = sum(1 for s in field_results.values() if s != "skip")

        status_icon = "✅" if qr_fields else "❌"
        print(
            f"[{tag_id}] {status_icon} "
            f"ts={int(ts)}ms | decoder={decoder} | "
            f"QR fields: {ok_count}/{total_count} ok | "
            f"barcode: {'✓' if barcode_val else '✗'} | "
            f"{row.get('product_name','')[:40]}"
        )

        per_tag_results.append({
            "tag_id":       tag_id,
            "timestamp_ms": ts,
            "product":      row.get("product_name", ""),
            "qr_decoded":   bool(qr_fields),
            "decoder":      decoder,
            "barcode_ok":   bool(barcode_val),
            "fields_ok":    ok_count,
            "fields_total": total_count,
            **{f"qr_{k}": v for k, v in qr_fields.items()},
        })

    # ── итоги ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ИТОГО")
    print(f"{'='*60}")
    n = stats["total"]
    if n == 0:
        print("Нет обработанных ценников.")
        return

    qr_rate  = stats["qr_decoded"] / n * 100
    bar_rate = stats["barcode_decoded"] / n * 100
    checked  = stats["field_ok"] + stats["field_miss"] + stats["field_wrong"]
    acc      = stats["field_ok"] / checked * 100 if checked else 0

    print(f"Ценников обработано : {n}")
    print(f"QR декодировано     : {stats['qr_decoded']}/{n}  ({qr_rate:.1f}%)")
    print(f"Штрихкод декодирован: {stats['barcode_decoded']}/{n}  ({bar_rate:.1f}%)")
    print(f"Точность QR-полей   : {stats['field_ok']}/{checked}  ({acc:.1f}%)")
    print(f"  ok={stats['field_ok']}  miss={stats['field_miss']}  wrong={stats['field_wrong']}  skip={stats['field_skip']}")

    # сохраняем детальный результат
    out_path = Path(csv_path).stem + "_qr_benchmark.csv"
    pd.DataFrame(per_tag_results).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nДетальный отчёт → {out_path}")

    # ── рекомендации ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ВЫВОДЫ")
    print(f"{'='*60}")
    if qr_rate >= 80:
        print("✅ QR читается хорошо — QR-first стратегия оправдана.")
        print("   OCR нужен только для product_name и fallback.")
    elif qr_rate >= 50:
        print("⚠️  QR читается в половине случаев — нужен aggressive preprocessing.")
        print("   Попробуй --save-crops и посмотри на проблемные кропы.")
    else:
        print("❌ QR плохо читается — фокусируемся на OCR как основном методе.")
        print("   QR оставляем как бонус там где читается.")


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QR benchmark для ценников Ленты")
    parser.add_argument("--csv",        required=True,  help="путь к CSV разметки")
    parser.add_argument("--video",      required=True,  help="путь к MP4 видео")
    parser.add_argument("--save-crops", action="store_true",
                        help="сохранить кропы ценников в папку debug_crops/")
    args = parser.parse_args()

    run_benchmark(args.csv, args.video, save_crops=args.save_crops)
