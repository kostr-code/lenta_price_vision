import marimo

__generated_with = "0.23.6"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo
    import cv2
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import pathlib

    return cv2, mo, mpatches, pathlib, pd, plt


@app.cell
def _(mo):
    mo.md("""
    ## Инспектор bbox — визуальная фильтрация разметки
    """)
    return


@app.cell
def _(mo):
    DATASETS = {
        "43_15": {
            "csv": "Данные/43_15/43_15.csv",
            "video": "Данные/43_15/43_15.mp4",
        },
        "25_12-20": {
            "csv": "Данные/25_12-20/25_12-20.csv",
            "video": "Данные/25_12-20/25_12-20.mp4",
        },
        "26_12-20": {
            "csv": "Данные/26_12-20/26_12-20.csv",
            "video": "Данные/26_12-20/26_12-20.mp4",
        },
    }
    dataset_picker = mo.ui.dropdown(
        list(DATASETS.keys()), value="43_15", label="Датасет"
    )
    return DATASETS, dataset_picker


@app.cell
def _(dataset_picker):
    dataset_picker
    return


@app.cell
def _(DATASETS, dataset_picker, pd):
    _cfg = DATASETS[dataset_picker.value]
    _df = pd.read_csv(_cfg["csv"], dtype=str)
    _df.columns = _df.columns.str.strip()

    for _col in ["x_min", "y_min", "x_max", "y_max"]:
        _df[_col] = _df[_col].str.replace(",", ".").astype(float)
    
    _df["frame_timestamp"] = pd.to_numeric(_df["frame_timestamp"], errors="coerce")
    all_df = _df
    return (all_df,)


@app.cell
def _(all_df, mo):
    _ts_vals = sorted(all_df["frame_timestamp"].dropna().unique().tolist())
    ts_picker = mo.ui.dropdown(
        {f"{int(ts)} ms": ts for ts in _ts_vals},
        value=f"{_ts_vals[0]} ms",
        label="Timestamp",
    )
    return (ts_picker,)


@app.cell
def _(ts_picker):
    ts_picker
    return


@app.cell
def _(all_df, ts_picker):
    ts_df = all_df[all_df["frame_timestamp"] == ts_picker.value].reset_index(drop=False)
    return (ts_df,)


@app.cell
def _(mo):
    scan_win = mo.ui.slider(0, 60, value=20, step=5, label="±N кадров (поиск резкого)")
    load_button = mo.ui.run_button(label="▶ Загрузить кадр и нарезать кропы")
    return load_button, scan_win


@app.cell
def _(load_button, mo, scan_win):
    mo.vstack([
        mo.md(
            "Ищет самый резкий кадр в окне ±N вокруг выбранного timestamp, "
            "рисует все bbox и нарезает кропы."
        ),
        mo.hstack([scan_win, load_button]),
    ])
    return


@app.cell
def _(DATASETS, cv2, dataset_picker, load_button, scan_win, ts_df, ts_picker):
    def _find_best_frame(video_path, ts_ms, n_frames):
        W_ORIG = 3840
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_ms = 1000.0 / fps
        best_var, best_frame = -1.0, None
        for i in range(-n_frames, n_frames + 1):
            ts = ts_ms + i * frame_ms
            cap.set(cv2.CAP_PROP_POS_MSEC, ts)
            ok, frame = cap.read()
            if not ok:
                continue
            rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            lap = cv2.Laplacian(cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if lap > best_var:
                best_var, best_frame = lap, rotated.copy()
        cap.release()
        return best_frame, best_var

    def _cut_crop(rotated, row):
        W_ORIG = 3840
        bx1 = int(row["y_min"])
        by1 = int(W_ORIG - 1 - row["x_max"])
        bx2 = int(row["y_max"])
        by2 = int(W_ORIG - 1 - row["x_min"])
        fh, fw = rotated.shape[:2]
        c = rotated[max(0, by1):min(fh, by2), max(0, bx1):min(fw, bx2)]
        return c if c.size > 0 else None

    if load_button.value:
        _cfg = DATASETS[dataset_picker.value]
        _frame, _var = _find_best_frame(
            _cfg["video"], float(ts_picker.value), scan_win.value
        )
        if _frame is not None:
            _crops = []
            for _, _row in ts_df.iterrows():
                _crops.append(_cut_crop(_frame, _row))
            frame_data = {"frame": _frame, "crops": _crops, "lap_var": _var}
        else:
            frame_data = None
    else:
        frame_data = None
    return (frame_data,)


@app.cell
def _(cv2, frame_data, mo, mpatches, plt, ts_df):
    def frame_widget(data, rows):
        if data is None:
            return mo.md("_Нажми **▶ Загрузить кадр** выше_")
        _rotated = data["frame"]
        _fig, _ax = plt.subplots(figsize=(18, 10))
        _ax.imshow(cv2.cvtColor(_rotated, cv2.COLOR_BGR2RGB))
        W_ORIG = 3840
        for _i, (_, _row) in enumerate(rows.iterrows()):
            _bx1 = int(_row["y_min"])
            _by1 = int(W_ORIG - 1 - _row["x_max"])
            _bx2 = int(_row["y_max"])
            _by2 = int(W_ORIG - 1 - _row["x_min"])
            _rect = mpatches.Rectangle(
                (_bx1, _by1), _bx2 - _bx1, _by2 - _by1,
                linewidth=2, edgecolor="lime", facecolor="none",
            )
            _ax.add_patch(_rect)
            _ax.text(_bx1 + 4, _by1 + 20, str(_i), color="yellow",
                     fontsize=10, fontweight="bold",
                     bbox=dict(facecolor="black", alpha=0.5, pad=1))
        _ax.set_title(
            f"Повёрнутый кадр  Laplacian={data['lap_var']:.1f}  |  {len(rows)} bbox",
            fontsize=12,
        )
        _ax.axis("off")
        plt.tight_layout()
        return _fig

    frame_widget(frame_data, ts_df)
    return


@app.cell
def _(cv2, frame_data, mo, plt, ts_df):
    def crops_grid_widget(data, rows):
        if data is None:
            return mo.md("_Загрузи кадр выше_")
        _crops = data["crops"]
        _valid = [(i, c, rows.iloc[i]) for i, c in enumerate(_crops) if c is not None]
        if not _valid:
            return mo.callout(mo.md("**Ни один кроп не получился**"), kind="warn")
        _N_COLS = 5
        _N_ROWS = (len(_valid) + _N_COLS - 1) // _N_COLS
        _fig, _axes = plt.subplots(_N_ROWS, _N_COLS, figsize=(4 * _N_COLS, 4 * _N_ROWS))
        _axes = _axes.flatten() if _N_ROWS > 1 or _N_COLS > 1 else [_axes]
        for _ax in _axes:
            _ax.axis("off")
        for _pos, (_i, _crop, _row) in enumerate(_valid):
            _ax = _axes[_pos]
            _ax.imshow(cv2.cvtColor(_crop, cv2.COLOR_BGR2RGB))
            _name = str(_row.get("product_name", ""))[:20]
            _ax.set_title(f"[{_i}] {_name}", fontsize=8)
            _ax.axis("off")
        plt.suptitle("Сетка кропов (номер соответствует bbox на кадре выше)", fontsize=12)
        plt.tight_layout()
        return _fig

    crops_grid_widget(frame_data, ts_df)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Выбери хорошие bbox
    "
        "Отметь строки которые содержат нормальный ценник с видимым QR. "
        "Используй кадр и сетку кропов выше как ориентир.
    """)
    return


@app.cell
def _(mo, ts_df):
    _display = ts_df[["index", "product_name", "qr_code_barcode", "frame_timestamp"]].copy()
    _display.columns = ["idx", "Продукт", "GT QR", "Timestamp"]
    selection_table = mo.ui.table(_display, selection="multi")
    return (selection_table,)


@app.cell
def _(selection_table):
    selection_table
    return


@app.cell
def _(mo, selection_table):
    _sel = selection_table.value
    (
        mo.callout(
            mo.md(f"**Выбрано строк: {len(_sel)}** — нажми кнопку ниже чтобы сохранить кропы"),
            kind="success",
        )
        if len(_sel) > 0
        else mo.md("_Выбери строки в таблице выше_")
    )
    return


@app.cell
def _(mo):
    save_button = mo.ui.run_button(label="▶ Сохранить выбранные кропы")
    save_button
    return (save_button,)


@app.cell
def _(
    cv2,
    dataset_picker,
    frame_data,
    mo,
    pathlib,
    pd,
    save_button,
    selection_table,
    ts_df,
):
    def save_widget(btn, sel, data, rows, ds_name):
        if not btn:
            return mo.md("_Нажми **▶ Сохранить** выше_")
        if not sel or data is None:
            return mo.callout(mo.md("**Нечего сохранять** — выбери строки и загрузи кадр"), kind="warn")
        _out = pathlib.Path("qr_crops") / ds_name / "good"
        _out.mkdir(parents=True, exist_ok=True)
        _meta_rows, _saved = [], []
        for _, _sel_row in sel.iterrows():
            _orig_idx = int(_sel_row["idx"])
            _pos_in_ts = ts_df[ts_df["index"] == _orig_idx].index
            if len(_pos_in_ts) == 0:
                continue
            _pos = _pos_in_ts[0]
            _crop = data["crops"][_pos]
            if _crop is None:
                continue
            _fname = f"{_orig_idx:03d}.jpg"
            cv2.imwrite(str(_out / _fname), _crop)
            _saved.append(_fname)
            _orig_row = ts_df.iloc[_pos]
            _meta_rows.append({
                "filename": _fname,
                "product_name": _orig_row.get("product_name", ""),
                "gt_qr": _orig_row.get("qr_code_barcode", ""),
            })
        pd.DataFrame(_meta_rows).to_csv(_out / "good_meta.csv", index=False)
        return mo.callout(
            mo.md(f"**Сохранено {len(_saved)} кропов** → `{_out}/`\n\n" +
                  "\n".join(f"- `{f}`" for f in _saved)),
            kind="success",
        )

    save_widget(
        save_button.value,
        selection_table.value,
        frame_data,
        ts_df,
        dataset_picker.value,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Unlabeled видео — YOLO детекция
    Нет CSV-разметки. Детектируем ценники YOLOv8 моделью.
    """)
    return


@app.cell
def _(mo):
    UNLABELED_VIDEOS = {
        "25_12-20": "Данные/Unlabeled/25_12-20.mp4",
        "26_12-20": "Данные/Unlabeled/26_12-20.mp4",
        "26_2-10":  "Данные/Unlabeled/26_2-10.mp4",
    }
    ul_video_picker = mo.ui.dropdown(
        list(UNLABELED_VIDEOS.keys()), value="26_12-20", label="Видео"
    )
    yolo_weights = mo.ui.text(
        placeholder="путь к .pt файлу, например yolo_pricetag.pt",
        label="Веса YOLOv8 (.pt)",
        full_width=True,
    )
    return UNLABELED_VIDEOS, ul_video_picker, yolo_weights


@app.cell
def _(mo, ul_video_picker, yolo_weights):
    mo.vstack([ul_video_picker, yolo_weights])
    return


@app.cell
def _(mo):
    ul_ts_slider = mo.ui.slider(0, 120_000, value=5000, step=500, label="Timestamp (ms)")
    ul_scan_win = mo.ui.slider(0, 60, value=20, step=5, label="±N кадров (поиск резкого)")
    ul_detect_btn = mo.ui.run_button(label="▶ Загрузить кадр + YOLO детекция")
    mo.vstack([
        mo.hstack([ul_ts_slider, ul_scan_win]),
        ul_detect_btn,
    ])
    return ul_detect_btn, ul_scan_win, ul_ts_slider


@app.cell
def _(
    UNLABELED_VIDEOS,
    cv2,
    mo,
    mpatches,
    plt,
    ul_detect_btn,
    ul_scan_win,
    ul_ts_slider,
    ul_video_picker,
    yolo_weights,
):
    def _find_sharpest(video_path, ts_ms, n_frames):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        step = 1000.0 / fps
        best_var, best_frame = -1.0, None
        for i in range(-n_frames, n_frames + 1):
            cap.set(cv2.CAP_PROP_POS_MSEC, ts_ms + i * step)
            ok, raw = cap.read()
            if not ok:
                continue
            rot = cv2.rotate(raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
            v = cv2.Laplacian(cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if v > best_var:
                best_var, best_frame = v, rot.copy()
        cap.release()
        return best_frame, best_var

    def _yolo_detect(frame, weights_path):
        try:
            from ultralytics import YOLO
            model = YOLO(weights_path)
            results = model(frame, verbose=False)
            boxes = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf})
            return boxes, None
        except ImportError:
            return [], "ultralytics не установлен"
        except Exception as e:
            return [], str(e)

    if ul_detect_btn.value:
        _vpath = UNLABELED_VIDEOS[ul_video_picker.value]
        _frame, _var = _find_sharpest(_vpath, float(ul_ts_slider.value), ul_scan_win.value)
        if _frame is None:
            ul_detect_result = mo.callout(mo.md("**Кадр не найден**"), kind="warn")
        else:
            _weights = yolo_weights.value.strip()
            if not _weights:
                ul_detect_result = mo.callout(
                    mo.md("**Укажи путь к .pt файлу весов** в поле выше"),
                    kind="info",
                )
            else:
                _boxes, _err = _yolo_detect(_frame, _weights)
                if _err:
                    ul_detect_result = mo.callout(mo.md(f"**YOLO ошибка:** {_err}"), kind="warn")
                else:
                    _fig, _ax = plt.subplots(figsize=(18, 10))
                    _ax.imshow(cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB))
                    for _i, _b in enumerate(_boxes):
                        _rect = mpatches.Rectangle(
                            (_b["x1"], _b["y1"]),
                            _b["x2"] - _b["x1"],
                            _b["y2"] - _b["y1"],
                            linewidth=2, edgecolor="cyan", facecolor="none",
                        )
                        _ax.add_patch(_rect)
                        _ax.text(
                            _b["x1"] + 4, _b["y1"] + 20,
                            f"[{_i}] {_b['conf']:.2f}",
                            color="yellow", fontsize=9, fontweight="bold",
                            bbox=dict(facecolor="black", alpha=0.5, pad=1),
                        )
                    _ax.set_title(
                        f"YOLO: {len(_boxes)} ценников | Laplacian={_var:.1f} | {ul_video_picker.value}",
                        fontsize=12,
                    )
                    _ax.axis("off")
                    plt.tight_layout()
                    ul_detect_result = _fig
    else:
        ul_detect_result = mo.md("_Нажми **▶ Загрузить кадр + YOLO детекция** выше_")

    ul_detect_result
    return (ul_detect_result,)


if __name__ == "__main__":
    app.run()
