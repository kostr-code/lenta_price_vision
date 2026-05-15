import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full", app_title="Price Tag Pipeline — E2E Test")


# ── 0. Imports ────────────────────────────────────────────────────────────────
@app.cell
def _():
    import sys
    import pathlib
    import base64

    import cv2
    import numpy as np
    import pandas as pd
    import marimo as mo

    # make pipeline/ importable from experiments/
    _here = pathlib.Path(__file__).parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    from pipeline import video as _vid, ocr as _ocr_mod, qr as _qr_mod, sr as _sr_mod
    from pipeline import parsers as _parsers

    return (
        base64,
        cv2,
        mo,
        np,
        pathlib,
        pd,
        sys,
        _vid,
        _ocr_mod,
        _qr_mod,
        _sr_mod,
        _parsers,
    )


# ── 1. Model loaders ──────────────────────────────────────────────────────────
@app.cell
def _(mo, _ocr_mod, _qr_mod, _sr_mod):
    mo.md("## 1. Загрузка моделей")
    return


@app.cell
def _(mo, _ocr_mod):
    try:
        ocr_model = _ocr_mod.load_ocr(use_gpu=False, lang="ru")
        _ocr_status = mo.callout(mo.md("**PaddleOCR**: OK (CPU, lang=ru)"), kind="success")
    except Exception as _e:
        ocr_model = None
        _ocr_status = mo.callout(mo.md(f"**PaddleOCR**: ошибка — `{_e}`"), kind="danger")
    _ocr_status
    return (ocr_model,)


@app.cell
def _(mo, _qr_mod):
    try:
        wechat = _qr_mod.load_wechat()
        _wechat_status = mo.callout(mo.md("**WeChat QR**: OK"), kind="success")
    except Exception as _e:
        wechat = None
        _wechat_status = mo.callout(mo.md(f"**WeChat QR**: недоступен — `{_e}`"), kind="warn")
    _wechat_status
    return (wechat,)


@app.cell
def _(mo, _sr_mod):
    use_esrgan = mo.ui.checkbox(label="Включить Real-ESRGAN ×4 (требует GPU, медленно)", value=False)
    use_esrgan
    return (use_esrgan,)


@app.cell
def _(mo, use_esrgan, _sr_mod):
    if use_esrgan.value:
        try:
            esrgan = _sr_mod.load_esrgan(gpu_id=0)
            _sr_status = mo.callout(mo.md("**Real-ESRGAN**: OK"), kind="success")
        except Exception as _e:
            esrgan = None
            _sr_status = mo.callout(mo.md(f"**Real-ESRGAN**: ошибка — `{_e}`"), kind="warn")
    else:
        esrgan = None
        _sr_status = mo.callout(mo.md("**Real-ESRGAN**: отключён"), kind="neutral")
    _sr_status
    return (esrgan,)


# ── 2. Dataset config UI ──────────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md("## 2. Выбор датасета")
    return


@app.cell
def _(mo):
    DATASETS = {
        "43_15": {
            "csv":   "Данные/43_15/43_15.csv",
            "video": "Данные/43_15/43_15.mp4",
        },
        "25_12-20": {
            "csv":   "Данные/25_12-20/25_12-20.csv",
            "video": "Данные/25_12-20/25_12-20.mp4",
        },
        "26_12-20": {
            "csv":   "Данные/26_12-20/26_12-20.csv",
            "video": "Данные/26_12-20/26_12-20.mp4",
        },
    }

    dataset_sel = mo.ui.dropdown(
        list(DATASETS.keys()), value="43_15", label="Датасет"
    )
    scan_slider = mo.ui.slider(5, 60, value=20, step=5, label="±N кадров для поиска резкости")
    limit_num = mo.ui.number(start=0, stop=500, step=1, value=0, label="Лимит строк (0 = все)")

    mo.hstack([dataset_sel, scan_slider, limit_num], gap="1rem")
    return DATASETS, dataset_sel, scan_slider, limit_num


# ── 3. Load CSV ───────────────────────────────────────────────────────────────
@app.cell
def _(mo, DATASETS, dataset_sel, limit_num, _vid, pathlib):
    _cfg = DATASETS[dataset_sel.value]
    _csv_path = pathlib.Path(_cfg["csv"])

    if not _csv_path.exists():
        df_labeled = None
        _info = mo.callout(mo.md(f"**CSV не найден:** `{_csv_path}`"), kind="danger")
    else:
        df_labeled = _vid.load_df(str(_csv_path))
        _n_lim = int(limit_num.value) if limit_num.value else len(df_labeled)
        df_labeled = df_labeled.head(_n_lim)
        _ts_uniq = df_labeled["frame_timestamp"].nunique() if "frame_timestamp" in df_labeled.columns else "?"
        _info = mo.callout(
            mo.md(
                f"**{dataset_sel.value}**: {len(df_labeled)} строк, "
                f"{_ts_uniq} уникальных timestamp  "
                f"→ `{_cfg['video']}`"
            ),
            kind="success",
        )
    _info
    return (df_labeled,)


# ── 4. Batch run ──────────────────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md("## 3. Запуск пайплайна")
    return


@app.cell
def _(mo):
    run_btn = mo.ui.run_button(label="▶ Запустить на всех кропах")
    run_btn
    return (run_btn,)


@app.cell
def _(
    mo,
    run_btn,
    df_labeled,
    DATASETS,
    dataset_sel,
    scan_slider,
    ocr_model,
    wechat,
    esrgan,
    cv2,
    np,
    _vid,
    _ocr_mod,
    _qr_mod,
    _sr_mod,
    _parsers,
    pathlib,
):
    if not run_btn.value or df_labeled is None:
        pipeline_rows = None
        pipeline_crops = {}  # row_idx → {"raw": img, "annotated": img, "qr_sub": img}
    else:
        _cfg = DATASETS[dataset_sel.value]
        _video_path = _cfg["video"]
        pipeline_rows = []
        pipeline_crops = {}

        for _pos, (_, _row) in enumerate(df_labeled.iterrows()):
            _ts = float(_row["frame_timestamp"])
            _gt_name  = str(_row.get("product_name", "")).strip()
            _gt_price = str(_row.get("price_card", "")).strip()
            _gt_qr    = str(_row.get("qr_code_barcode", "")).strip()

            # 1. Find best frame
            _frame, _lap = _vid.find_best_frame(_video_path, _ts, int(scan_slider.value))
            if _frame is None:
                continue

            _crop = _vid.cut_crop_from_row(_frame, _row)
            if _crop is None:
                continue

            _raw_h, _raw_w = _crop.shape[:2]

            # 2. Upscale (ESRGAN or Lanczos×4)
            if esrgan is not None:
                _img_ocr = _sr_mod.upscale_safe(_crop, esrgan, scale=4)
            else:
                _img_ocr = cv2.resize(_crop, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)

            _h_up, _w_up = _img_ocr.shape[:2]

            # 3. QR decode on original-resolution sub-crop (before upscale distortion)
            _qr_sub = _qr_mod.cut_qr_subcrop(_crop)
            _qr_decoded: list[str] = []
            if _qr_sub is not None:
                _qr_decoded = _qr_mod.decode_qr(_qr_sub, wechat=wechat)

            # 4. Zonal OCR → OCRLine list → parse all fields
            if ocr_model is not None:
                _lines = _ocr_mod.ocr_zoned(ocr_model, _img_ocr)
                _annotated = _ocr_mod.annotate_lines(_img_ocr, _lines)
            else:
                _lines = []
                _annotated = _img_ocr.copy()

            # 5. Parse all 29 fields
            _fields = _parsers.parse_fields(_lines, _qr_decoded, crop_bgr=_crop)

            # 6. GT name match (word overlap)
            _name_words = set(_gt_name.lower().split()) if _gt_name else set()
            _ocr_name = _fields.get("product_name", "").lower()
            _name_match = (
                sum(1 for w in _name_words if w in _ocr_name) / max(len(_name_words), 1)
                if _name_words else 0.0
            )

            pipeline_rows.append({
                "row_idx":         _pos,
                "frame_ts_ms":     _ts,
                "lap_var":         round(_lap, 1),
                "raw_w":           _raw_w,
                "raw_h":           _raw_h,
                "gt_product_name": _gt_name,
                "gt_price_card":   _gt_price,
                "gt_qr":           _gt_qr,
                "qr_ok":           bool(_qr_decoded),
                "has_ocr":         bool(_lines),
                "name_match":      round(_name_match, 3),
                **_fields,
            })

            # raw OCR text zones (for inspector display)
            _zones_raw = _ocr_mod.extract_text_zones(
                ocr_model.ocr(_img_ocr, cls=True) if ocr_model else None,
                _h_up, _w_up
            ) if ocr_model else {"top": [], "mid": [], "bottom": []}

            pipeline_crops[_pos] = {
                "raw":       _crop,
                "annotated": _annotated,
                "qr_sub":    _qr_sub,
                "up_w":      _w_up,
                "up_h":      _h_up,
                "zones":     _zones_raw,
                "lines":     _lines,
            }

    return (pipeline_rows, pipeline_crops)


# ── 5. Results table ──────────────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md("## 4. Результаты")
    return


@app.cell
def _(mo, pipeline_rows, pd):
    if pipeline_rows is None:
        mo.md("_Нажми ▶ Запустить выше_")
    elif len(pipeline_rows) == 0:
        mo.callout(mo.md("Нет результатов — проверь пути к файлам"), kind="warn")
    else:
        _df = pd.DataFrame(pipeline_rows)
        _total = len(_df)
        _has_ocr = int(_df["has_ocr"].sum())
        _qr_ok   = int(_df["qr_ok"].sum())
        _bc_ok   = int(_df["barcode_ok"].sum())
        _avg_match = _df["name_match"].mean()

        _summary = mo.callout(
            mo.md(
                f"**Итого**: {_total} ценников | "
                f"OCR: {_has_ocr}/{_total} ({100*_has_ocr//_total}%) | "
                f"QR: {_qr_ok}/{_total} | "
                f"Штрихкод: {_bc_ok}/{_total} | "
                f"name_match: {_avg_match:.0%}"
            ),
            kind="success" if _has_ocr > 0 else "warn",
        )

        _show_cols = [
            "row_idx", "gt_product_name", "product_name", "name_match",
            "gt_price_card", "price_card", "price_default",
            "barcode", "id_sku", "print_datetime", "code",
            "color", "special_symbols",
            "qr_ok", "has_ocr", "lap_var",
        ]
        _show_cols = [c for c in _show_cols if c in _df.columns]

        mo.vstack([_summary, mo.ui.table(_df[_show_cols])])
    return


# ── 6. Single crop inspector ──────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md("## 5. Инспектор кропа")
    return


@app.cell
def _(mo, pipeline_rows):
    _max_idx = max((r["row_idx"] for r in pipeline_rows), default=0) if pipeline_rows else 0
    inspect_idx = mo.ui.number(
        start=0, stop=_max_idx, step=1, value=0, label="Индекс строки"
    )
    inspect_idx
    return (inspect_idx,)


@app.cell
def _(mo, pipeline_rows, pipeline_crops, inspect_idx, cv2, base64, np):
    def _img_to_b64(img_bgr: "np.ndarray", max_w: int = 600) -> str:
        h, w = img_bgr.shape[:2]
        if w > max_w:
            img_bgr = cv2.resize(img_bgr, (max_w, int(h * max_w / w)))
        _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    if not pipeline_rows or pipeline_crops is None:
        mo.md("_Нет данных — сначала запусти пайплайн_")
    else:
        _idx = int(inspect_idx.value)
        _row = next((r for r in pipeline_rows if r["row_idx"] == _idx), None)
        _crops = pipeline_crops.get(_idx)

        if _row is None or _crops is None:
            mo.md(f"_Индекс {_idx} не найден_")
        else:
            _panels = []

            _panels.append(mo.md(f"**Сырой кроп** ({_crops['raw'].shape[1]}×{_crops['raw'].shape[0]})"))
            _panels.append(mo.Html(f'<img src="{_img_to_b64(_crops["raw"], 400)}" style="max-width:100%">'))

            _panels.append(mo.md(f"**Аннотированный** ({_crops['up_w']}×{_crops['up_h']})"))
            _panels.append(mo.Html(f'<img src="{_img_to_b64(_crops["annotated"], 600)}" style="max-width:100%">'))

            if _crops["qr_sub"] is not None:
                _qr_big = cv2.resize(_crops["qr_sub"], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
                _panels.append(mo.md(f"**QR sub-crop** ×4"))
                _panels.append(mo.Html(f'<img src="{_img_to_b64(_qr_big, 300)}" style="max-width:100%">'))

            def _fld(key: str) -> str:
                return str(_row.get(key, "") or "—")

            _info = mo.md(
                f"**GT name:** {_row['gt_product_name'][:60]}  \n"
                f"**OCR name:** {_fld('product_name')[:60]}  \n"
                f"**name_match:** {_row['name_match']:.0%}  \n\n"
                f"**price_card:** GT={_row['gt_price_card']}  OCR={_fld('price_card')}  \n"
                f"**price_default:** {_fld('price_default')}  \n"
                f"**price_discount:** {_fld('price_discount')}  \n"
                f"**discount_amount:** {_fld('discount_amount')}  \n\n"
                f"**barcode:** {_fld('barcode')}  \n"
                f"**id_sku:** {_fld('id_sku')}  \n"
                f"**print_datetime:** {_fld('print_datetime')}  \n"
                f"**code:** {_fld('code')}  \n"
                f"**color:** {_fld('color')}  \n"
                f"**special_symbols:** {_fld('special_symbols')}  \n"
                f"**additional_info:** {_fld('additional_info')}  \n\n"
                f"**QR decoded:** {_fld('qr_code_barcode')}  \n"
                f"**has_ocr:** {_row['has_ocr']}  lap={_row['lap_var']}"
            )

            mo.hstack([
                mo.vstack(_panels),
                _info,
            ], gap="1rem")
    return


# ── 7. QR experiments panel ───────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md("## 6. QR эксперименты (выбранный кроп)")
    return


@app.cell
def _(mo):
    qr_run_btn = mo.ui.run_button(label="▶ Попробовать все QR методы")
    qr_run_btn
    return (qr_run_btn,)


@app.cell
def _(
    mo,
    qr_run_btn,
    pipeline_crops,
    inspect_idx,
    wechat,
    esrgan,
    cv2,
    np,
    pd,
    _qr_mod,
    _sr_mod,
):
    if not qr_run_btn.value or pipeline_crops is None:
        mo.md("_Нажми ▶ Попробовать все QR методы_")
    else:
        _idx = int(inspect_idx.value)
        _crops = pipeline_crops.get(_idx)

        if _crops is None or _crops["qr_sub"] is None:
            mo.callout(mo.md("QR sub-crop не найден для этого кропа"), kind="warn")
        else:
            _qr_sub = _crops["qr_sub"]
            _gray = cv2.cvtColor(_qr_sub, cv2.COLOR_BGR2GRAY)

            def _try_decode(img_bgr, label: str) -> dict:
                _found = _qr_mod.decode_qr(img_bgr, wechat=wechat)
                return {"Метод": label, "Декодировано": " | ".join(_found) if _found else "❌"}

            _methods = [
                _try_decode(_qr_sub, "Оригинал"),
                _try_decode(
                    cv2.cvtColor(cv2.resize(_gray, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_GRAY2BGR),
                    "×2 resize (Lanczos)"
                ),
                _try_decode(
                    cv2.cvtColor(cv2.resize(_gray, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_GRAY2BGR),
                    "×4 resize (Lanczos)"
                ),
            ]

            # CLAHE + adaptive threshold
            _clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            _clahe_img = cv2.cvtColor(_clahe.apply(_gray), cv2.COLOR_GRAY2BGR)
            _methods.append(_try_decode(_clahe_img, "CLAHE"))

            _adapt = cv2.adaptiveThreshold(
                _gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            _methods.append(_try_decode(cv2.cvtColor(_adapt, cv2.COLOR_GRAY2BGR), "Adaptive threshold"))

            # ESRGAN ×4
            if esrgan is not None:
                _esr = _sr_mod.upscale_safe(_qr_sub, esrgan, scale=4)
                _methods.append(_try_decode(_esr, "Real-ESRGAN ×4"))
            else:
                _methods.append({"Метод": "Real-ESRGAN ×4", "Декодировано": "_(отключён)_"})

            _methods.append({"Метод": "ADNet (TODO)", "Декодировано": "_(не реализован)_"})

            _df_qr = pd.DataFrame(_methods)
            _ok_count = sum(1 for r in _methods if r["Декодировано"] not in ("❌", "_(отключён)_", "_(не реализован)_"))

            mo.vstack([
                mo.callout(
                    mo.md(f"Успешных методов: **{_ok_count}/{len(_methods)}**"),
                    kind="success" if _ok_count > 0 else "warn",
                ),
                mo.ui.table(_df_qr),
            ])
    return


# ── 8. Summary metrics ────────────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md("## 7. Итоговые метрики")
    return


@app.cell
def _(mo, pipeline_rows, pd):
    if not pipeline_rows:
        mo.md("_Нет данных_")
    else:
        _df = pd.DataFrame(pipeline_rows)
        _total = len(_df)
        _metrics = {
            "Всего ценников":        _total,
            "OCR что-то нашёл":      f"{int(_df['has_ocr'].sum())}/{_total}  ({100*int(_df['has_ocr'].sum())//_total}%)",
            "QR декодирован":         f"{int(_df['qr_ok'].sum())}/{_total}",
            "Линейный штрихкод":      f"{int(_df['barcode_ok'].sum())}/{_total}",
            "avg name_match":         f"{_df['name_match'].mean():.0%}",
            "avg lap_var (резкость)": f"{_df['lap_var'].mean():.0f}",
        }
        _rows = [{"Метрика": k, "Значение": v} for k, v in _metrics.items()]
        mo.ui.table(pd.DataFrame(_rows))
    return


if __name__ == "__main__":
    app.run()
