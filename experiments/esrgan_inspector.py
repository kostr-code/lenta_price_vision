import marimo

__generated_with = "0.23.6"
app = marimo.App(width="wide")


@app.cell
def _():
    import pathlib
    import urllib.request

    import cv2
    import marimo as mo
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import zxingcpp
    from pyzbar.pyzbar import decode as pyzbar_decode

    return (
        cv2,
        mo,
        mpatches,
        np,
        pathlib,
        pd,
        plt,
        pyzbar_decode,
        urllib,
        zxingcpp,
    )


@app.cell
def _(cv2, np, pathlib, pd, pyzbar_decode, zxingcpp):
    W_ORIG = 3840

    DATASETS = {
        "43_15": {"csv": "Данные/43_15/43_15.csv", "video": "Данные/43_15/43_15.mp4"},
        "25_12-20": {
            "csv": "Данные/25_12-20/25_12-20.csv",
            "video": "Данные/25_12-20/25_12-20.mp4",
        },
        "26_12-20": {
            "csv": "Данные/26_12-20/26_12-20.csv",
            "video": "Данные/26_12-20/26_12-20.mp4",
        },
    }

    WECHAT_MODEL_DIR = pathlib.Path("wechat_models")
    ESRGAN_MODEL_PTH = pathlib.Path("models/RealESRGAN_x4plus.pth")
    ESRGAN_MODEL_URL = (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    )

    def load_df(csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path, dtype=str)
        df.columns = df.columns.str.strip()
        for col in ["x_min", "y_min", "x_max", "y_max"]:
            df[col] = df[col].str.replace(",", ".").astype(float)
        df["frame_timestamp"] = pd.to_numeric(df["frame_timestamp"], errors="coerce")
        return df

    def find_best_frame(video_path: str, ts_ms: float, n: int = 20) -> tuple[np.ndarray | None, float]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        step = 1000.0 / fps
        best_var, best_frame = -1.0, None
        for i in range(-n, n + 1):
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

    def cut_crop(frame: np.ndarray, row: pd.Series) -> np.ndarray | None:
        bx1 = int(row["y_min"])
        by1 = int(W_ORIG - 1 - row["x_max"])
        bx2 = int(row["y_max"])
        by2 = int(W_ORIG - 1 - row["x_min"])
        fh, fw = frame.shape[:2]
        c = frame[max(0, by1) : min(fh, by2), max(0, bx1) : min(fw, bx2)]
        return c if c.size > 0 else None

    def cut_region(crop: np.ndarray, x_start_frac: float, y_end_frac: float) -> np.ndarray | None:
        h, w = crop.shape[:2]
        sub = crop[0 : int(h * y_end_frac), int(w * x_start_frac) :]
        return sub if sub.size > 0 else None

    def decode_qr(img_bgr: np.ndarray, wechat: object = None) -> list[str]:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        found = set()
        for r in pyzbar_decode(gray):
            if r.data:
                found.add(r.data.decode())
        for r in zxingcpp.read_barcodes(gray):
            if r.text:
                found.add(r.text)
        if wechat is not None:
            data, _ = wechat.detectAndDecode(img_bgr)
            for d in data:
                if d:
                    found.add(d)
        gray2 = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        for r in pyzbar_decode(gray2):
            if r.data:
                found.add(r.data.decode())
        for r in zxingcpp.read_barcodes(gray2):
            if r.text:
                found.add(r.text)
        return sorted(found)

    return (
        DATASETS,
        ESRGAN_MODEL_PTH,
        ESRGAN_MODEL_URL,
        WECHAT_MODEL_DIR,
        cut_crop,
        cut_region,
        decode_qr,
        find_best_frame,
        load_df,
    )


@app.cell
def _(mo):
    mo.md("""
    ## ESRGAN Inspector — До и После
    """)
    return


@app.cell
def _():
    # DANGER - HOTFIX - see: https://github.com/xinntao/Real-ESRGAN/issues/768
    import sys
    import types
    from torchvision.transforms.functional import rgb_to_grayscale

    # Create a module for `torchvision.transforms.functional_tensor`
    functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
    functional_tensor.rgb_to_grayscale = rgb_to_grayscale

    # Add this module to sys.modules so other imports can access it
    sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor
    return


@app.cell
def _(mo):
    load_btn = mo.ui.run_button(label="▶ Загрузить WeChat + Real-ESRGAN")
    mo.vstack(
        [
            mo.md("Загружает WeChat NN-детектор и Real-ESRGAN ×4 (модель ~67 MB, только GPU)."),
            load_btn,
        ]
    )
    return (load_btn,)


@app.cell
def _(
    ESRGAN_MODEL_PTH,
    ESRGAN_MODEL_URL,
    WECHAT_MODEL_DIR,
    cv2,
    load_btn,
    mo,
    pathlib,
    urllib,
):
    def _load_wechat():
        _files = ["detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel"]
        _base = "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/"
        WECHAT_MODEL_DIR.mkdir(exist_ok=True)
        for _f in _files:
            _p = WECHAT_MODEL_DIR / _f
            if not _p.exists():
                urllib.request.urlretrieve(_base + _f, _p)
        try:
            det = cv2.wechat_qrcode_WeChatQRCode(
                str(WECHAT_MODEL_DIR / "detect.prototxt"),
                str(WECHAT_MODEL_DIR / "detect.caffemodel"),
                str(WECHAT_MODEL_DIR / "sr.prototxt"),
                str(WECHAT_MODEL_DIR / "sr.caffemodel"),
            )
            return det, "NN-режим (детекция + SRQI)"
        except Exception as e:
            return cv2.wechat_qrcode_WeChatQRCode(), f"базовый ({e})"

    def _load_esrgan():
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        pathlib.Path("models").mkdir(exist_ok=True)
        if not ESRGAN_MODEL_PTH.exists():
            urllib.request.urlretrieve(ESRGAN_MODEL_URL, ESRGAN_MODEL_PTH)
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        return RealESRGANer(
            scale=4,
            model_path=str(ESRGAN_MODEL_PTH),
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
            gpu_id=0,
        )

    if load_btn.value:
        _wechat, _wechat_mode = _load_wechat()
        try:
            _esrgan = _load_esrgan()
            _esrgan_status = mo.callout(mo.md("Real-ESRGAN: **OK** (GPU FP16)"), kind="success")
        except Exception as _e:
            _esrgan = None
            _esrgan_status = mo.callout(mo.md(f"Real-ESRGAN: ошибка — `{_e}`"), kind="warn")
        wechat_det = _wechat
        esrgan_up = _esrgan
        _msg = mo.vstack(
            [
                mo.callout(mo.md(f"WeChat: **{_wechat_mode}**"), kind="success"),
                _esrgan_status,
            ]
        )
    else:
        wechat_det = None
        esrgan_up = None
        _msg = mo.md("_Нажми ▶ выше — модели нужны для ESRGAN и WeChat_")
    _msg
    return esrgan_up, wechat_det


@app.cell
def _(DATASETS, mo):
    ds_picker = mo.ui.dropdown(list(DATASETS.keys()), value="43_15", label="Датасет")
    ds_picker
    return (ds_picker,)


@app.cell
def _(DATASETS, ds_picker, load_df):
    df_current = load_df(DATASETS[ds_picker.value]["csv"])
    return (df_current,)


@app.cell
def _(df_current, mo):
    _disp = df_current[
        [
            "frame_timestamp",
            "product_name",
            "qr_code_barcode",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
        ]
    ].copy()
    _disp.insert(0, "idx", range(len(_disp)))
    row_table = mo.ui.table(_disp, selection="single", label="Выбери строку ценника")
    row_table
    return (row_table,)


@app.cell
def _(df_current, mo, row_table):
    _sel = row_table.value
    if len(_sel) > 0:
        _idx = int(_sel.iloc[0]["idx"])
        selected_row = df_current.iloc[_idx]
        _w = float(selected_row["y_max"]) - float(selected_row["y_min"])
        _h = float(selected_row["x_max"]) - float(selected_row["x_min"])
        _info = mo.callout(
            mo.md(
                f"**#{_idx}** {str(selected_row.get('product_name', ''))[:55]}  \n"
                f"GT QR: `{selected_row.get('qr_code_barcode', '')}`  \n"
                f"ts={float(selected_row['frame_timestamp']):.0f} ms  |  "
                f"bbox кроп (до поворота): **{_w:.0f}×{_h:.0f} px**"
            ),
            kind="info",
        )
    else:
        selected_row = None
        _info = mo.md("_Выбери строку выше_")
    _info
    return (selected_row,)


@app.cell
def _(mo):
    scan_slider = mo.ui.slider(5, 40, value=20, step=5, label="±N кадров (поиск резкого)")
    load_frame_btn = mo.ui.run_button(label="▶ Загрузить кадр")
    mo.hstack([scan_slider, load_frame_btn])
    return load_frame_btn, scan_slider


@app.cell
def _(
    DATASETS,
    cut_crop,
    ds_picker,
    find_best_frame,
    load_frame_btn,
    mo,
    scan_slider,
    selected_row,
):
    if load_frame_btn.value and selected_row is not None:
        _ts = float(selected_row["frame_timestamp"])
        _frame, _lap = find_best_frame(
            DATASETS[ds_picker.value]["video"],
            _ts,
            scan_slider.value,
        )
        if _frame is not None:
            _crop = cut_crop(_frame, selected_row)
            frame_data = {"frame": _frame, "crop": _crop, "lap": _lap, "ts": _ts}
        else:
            frame_data = None
    else:
        frame_data = None
    (
        mo.callout(
            mo.md(
                f"Кадр загружен  |  Laplacian={frame_data['lap']:.0f}  |  ts={frame_data['ts']:.0f}ms"
            ),
            kind="success",
        )
        if frame_data is not None
        else mo.md("_Выбери строку → нажми ▶ Загрузить кадр_")
    )
    return (frame_data,)


@app.cell
def _(cv2, frame_data, mo, mpatches, plt, selected_row):
    def _show_frame_crop(data: dict | None, row: object) -> object:
        if data is None or row is None:
            return mo.md("_Нет данных — загрузи кадр выше_")
        crop = data["crop"]
        if crop is None:
            return mo.callout(mo.md("**Кроп пустой** — bbox за пределами кадра"), kind="warn")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        # Полный повёрнутый кадр
        ax0 = axes[0]
        ax0.imshow(cv2.cvtColor(data["frame"], cv2.COLOR_BGR2RGB))
        _W = 3840
        bx1 = int(row["y_min"])
        by1 = int(_W - 1 - row["x_max"])
        bx2 = int(row["y_max"])
        by2 = int(_W - 1 - row["x_min"])
        ax0.add_patch(
            mpatches.Rectangle(
                (bx1, by1),
                bx2 - bx1,
                by2 - by1,
                linewidth=3,
                edgecolor="lime",
                facecolor="none",
            )
        )
        ax0.set_title(f"Повёрнутый кадр  Laplacian={data['lap']:.0f}", fontsize=11)
        ax0.axis("off")
        # Кроп ценника
        h, w = crop.shape[:2]
        axes[1].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Кроп ценника  {w}×{h} px", fontsize=11)
        axes[1].axis("off")
        plt.tight_layout()
        return fig

    _show_frame_crop(frame_data, selected_row)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Настрой QR-зону

    QR находится в **правом верхнем углу** ценника.
    Зелёный прямоугольник на кропе показывает текущую зону.

    - **x_start** — левая граница: 0.60 значит «правые 40% кропа»
    - **y_end** — нижняя граница: 0.42 значит «верхние 42% кропа»
    """)
    return


@app.cell
def _(mo):
    x_start = mo.ui.slider(
        0.0, 1.0, value=0.60, step=0.05, label="x_start — левый край QR (доля ширины кропа)"
    )
    y_end = mo.ui.slider(
        0.0, 1.0, value=0.42, step=0.05, label="y_end   — нижний край QR (доля высоты кропа)"
    )
    mo.vstack([x_start, y_end])
    return x_start, y_end


@app.cell
def _(cut_region, cv2, frame_data, mo, plt, x_start, y_end):
    def _show_qr_preview(data: dict | None, xs: float, ye: float) -> object:
        if data is None or data["crop"] is None:
            return mo.md("_Нет кропа_")
        crop = data["crop"]
        h, w = crop.shape[:2]
        qr_sub = cut_region(crop, xs, ye)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Кроп с нанесённой зелёной рамкой QR-зоны
        annotated = crop.copy()
        rx1 = int(w * xs)
        ry2 = int(h * ye)
        cv2.rectangle(annotated, (rx1, 0), (w - 1, ry2), (0, 255, 0), 2)
        cv2.putText(
            annotated, "QR region", (rx1 + 3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
        axes[0].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Кроп с QR-зоной  {w}×{h}px", fontsize=11)
        axes[0].axis("off")

        # Сырой QR sub-crop
        if qr_sub is not None:
            qh, qw = qr_sub.shape[:2]
            axes[1].imshow(cv2.cvtColor(qr_sub, cv2.COLOR_BGR2RGB))
            axes[1].set_title(
                f"QR sub-crop  {qw}×{qh}px\n(→ ×4 = {qw * 4}×{qh * 4}px)", fontsize=11
            )
        else:
            axes[1].text(0.5, 0.5, "Пустой sub-crop", ha="center", transform=axes[1].transAxes)
        axes[1].axis("off")

        plt.suptitle(f"x_start={xs:.0%}  y_end={ye:.0%}", fontsize=11, y=1.01)
        plt.tight_layout()
        return fig

    _show_qr_preview(frame_data, x_start.value, y_end.value)
    return


@app.cell
def _(mo):
    esrgan_btn = mo.ui.run_button(label="▶ Применить ESRGAN и декодировать")
    mo.vstack(
        [
            mo.md(
                "Запускает **Lanczos ×4** и **Real-ESRGAN ×4** на QR-зоне и пробует все декодеры:"
            ),
            esrgan_btn,
        ]
    )
    return (esrgan_btn,)


@app.cell
def _(
    cut_region,
    cv2,
    decode_qr,
    esrgan_btn,
    esrgan_up,
    frame_data,
    mo,
    plt,
    wechat_det,
    x_start,
    y_end,
):
    def _compare(data: dict | None, xs: float, ye: float, esrgan: object, wechat: object, triggered: bool) -> object:
        if not triggered:
            return mo.md("_Нажми ▶ Применить ESRGAN_")
        if data is None or data["crop"] is None:
            return mo.callout(mo.md("Загрузи кадр сначала"), kind="warn")

        crop = data["crop"]
        qr_sub = cut_region(crop, xs, ye)
        if qr_sub is None:
            return mo.callout(mo.md("QR sub-crop пустой — поправь слайдеры"), kind="warn")
        qh, qw = qr_sub.shape[:2]

        panels = []

        # 1. Сырой QR sub-crop
        raw_dec = decode_qr(qr_sub, wechat)
        panels.append(
            {
                "img": cv2.cvtColor(qr_sub, cv2.COLOR_BGR2RGB),
                "title": f"Сырой  {qw}×{qh}px",
                "dec": raw_dec,
            }
        )

        # 2. Lanczos ×4
        lanczos = cv2.resize(qr_sub, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)
        lan_dec = decode_qr(lanczos, wechat)
        panels.append(
            {
                "img": cv2.cvtColor(lanczos, cv2.COLOR_BGR2RGB),
                "title": f"Lanczos ×4  {lanczos.shape[1]}×{lanczos.shape[0]}px",
                "dec": lan_dec,
            }
        )

        # 3. ESRGAN ×4
        if esrgan is not None:
            try:
                up, _ = esrgan.enhance(qr_sub, outscale=4)
                up_dec = decode_qr(up, wechat)
                panels.append(
                    {
                        "img": cv2.cvtColor(up, cv2.COLOR_BGR2RGB),
                        "title": f"ESRGAN ×4  {up.shape[1]}×{up.shape[0]}px",
                        "dec": up_dec,
                    }
                )
            except Exception as e:
                panels.append(
                    {
                        "img": panels[1]["img"],  # placeholder
                        "title": f"ESRGAN ошибка: {e}",
                        "dec": [],
                    }
                )
        else:
            panels.append(
                {
                    "img": panels[1]["img"],
                    "title": "ESRGAN не загружен\n(нажми ▶ Загрузить вверху)",
                    "dec": [],
                }
            )

        # Plot
        ncols = len(panels)
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 7))
        if ncols == 1:
            axes = [axes]
        for ax, p in zip(axes, panels):
            ax.imshow(p["img"])
            status = "✅ " + " | ".join(p["dec"]) if p["dec"] else "❌ не декодировано"
            ax.set_title(f"{p['title']}\n{status}", fontsize=10, pad=6)
            ax.axis("off")
        plt.suptitle(
            f"QR-зона: x≥{xs:.0%}  y≤{ye:.0%}  |  sub-crop: {qw}×{qh}px",
            fontsize=12,
            y=1.01,
        )
        plt.tight_layout()
        return fig

    _compare(frame_data, x_start.value, y_end.value, esrgan_up, wechat_det, esrgan_btn.value)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Полный кроп ценника — до и после ESRGAN (для OCR)
    """)
    return


@app.cell
def _(mo):
    full_esrgan_btn = mo.ui.run_button(label="▶ ESRGAN на полный кроп (медленнее)")
    full_esrgan_btn
    return (full_esrgan_btn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ### Свои изображения — ESRGAN без CSV

    Загружай кропы напрямую: одно фото или целая папка.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    manual_upload = mo.ui.file(
        filetypes=[".jpg", ".jpeg", ".png", ".bmp"],
        multiple=True,
        kind="area",
        label="Перетащи изображения сюда",
    )
    manual_folder = mo.ui.text(
        placeholder="или путь к папке: results/ocr/raw_crops",
        label="...или папка",
        full_width=True,
    )
    mo.vstack([manual_upload, manual_folder])
    return manual_folder, manual_upload


@app.cell(hide_code=True)
def _(cv2, manual_folder, manual_upload, mo, np, pathlib):
    def _load_upload(files):
        out = []
        for f in files:
            buf = np.frombuffer(f.contents, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                out.append({"name": f.name, "img": img})
        return out

    def _load_folder(path_str):
        out = []
        p = pathlib.Path(path_str.strip())
        if not p.is_dir():
            return out
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                img = cv2.imread(str(f))
                if img is not None:
                    out.append({"name": f.name, "img": img})
        return out

    if manual_upload.value:
        manual_images = _load_upload(manual_upload.value)
    elif manual_folder.value.strip():
        manual_images = _load_folder(manual_folder.value)
    else:
        manual_images = []

    mo.callout(
        mo.md(f"Загружено: **{len(manual_images)}** изображений"),
        kind="success" if manual_images else "info",
    )
    return (manual_images,)


@app.cell(hide_code=True)
def _(manual_images, mo):
    if manual_images:
        _names = [img["name"] for img in manual_images]
        manual_img_picker = mo.ui.dropdown(_names, value=_names[0], label="Изображение")
        _widget = manual_img_picker
    else:
        manual_img_picker = None
        _widget = mo.md("_Загрузи файлы выше_")
    _widget
    return (manual_img_picker,)


@app.cell(hide_code=True)
def _(manual_images, manual_img_picker):
    if manual_img_picker is not None and manual_images:
        _idx = next(
            (i for i, m in enumerate(manual_images) if m["name"] == manual_img_picker.value), 0
        )
        selected_manual = manual_images[_idx]
    else:
        selected_manual = None
    return (selected_manual,)


@app.cell(hide_code=True)
def _(cv2, mo, plt, selected_manual):
    def _preview_manual(img_data: dict | None) -> object:
        if img_data is None:
            return mo.md("_Нет изображения_")
        img = img_data["img"]
        h, w = img.shape[:2]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{img_data['name']}  {w}×{h}px", fontsize=11)
        ax.axis("off")
        plt.tight_layout()
        return fig

    _preview_manual(selected_manual)
    return


@app.cell(hide_code=True)
def _(mo):
    enhance_manual_btn = mo.ui.run_button(label="▶ ESRGAN + декодировать")
    mo.vstack([
        mo.md("Применяет Lanczos×4 и ESRGAN×4 (если загружен), пробует все декодеры:"),
        enhance_manual_btn,
    ])
    return (enhance_manual_btn,)


@app.cell(hide_code=True)
def _(
    cv2,
    decode_qr,
    enhance_manual_btn,
    esrgan_up,
    mo,
    plt,
    selected_manual,
    wechat_det,
):
    def _enhance_and_decode(img_data: dict | None, esrgan: object, wechat: object, triggered: bool) -> object:
        if not triggered:
            return mo.md("_Нажми ▶ ESRGAN + декодировать_")
        if img_data is None:
            return mo.callout(mo.md("Загрузи и выбери изображение сначала"), kind="warn")
        img = img_data["img"]
        h, w = img.shape[:2]
        panels = []
        raw_dec = decode_qr(img, wechat)
        panels.append({"img": cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                       "title": f"Оригинал  {w}×{h}px", "dec": raw_dec})
        lan = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)
        lan_dec = decode_qr(lan, wechat)
        panels.append({"img": cv2.cvtColor(lan, cv2.COLOR_BGR2RGB),
                       "title": f"Lanczos×4  {lan.shape[1]}×{lan.shape[0]}px", "dec": lan_dec})
        if esrgan is not None:
            try:
                up, _ = esrgan.enhance(img, outscale=4)
                up_dec = decode_qr(up, wechat)
                panels.append({"img": cv2.cvtColor(up, cv2.COLOR_BGR2RGB),
                               "title": f"ESRGAN×4  {up.shape[1]}×{up.shape[0]}px", "dec": up_dec})
            except Exception as _e:
                panels.append({"img": panels[1]["img"], "title": f"ESRGAN err: {_e}", "dec": []})
        ncols = len(panels)
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 7))
        if ncols == 1:
            axes = [axes]
        for ax, p in zip(axes, panels):
            ax.imshow(p["img"])
            dec_str = " | ".join(p["dec"]) if p["dec"] else "не декодировано"
            status = ("✅ " if p["dec"] else "❌ ") + dec_str
            title_line = p["title"] + chr(10) + status
            ax.set_title(title_line, fontsize=10, pad=6)
            ax.axis("off")
        plt.tight_layout()
        return fig

    _enhance_and_decode(selected_manual, esrgan_up, wechat_det, enhance_manual_btn.value)
    return


@app.cell(hide_code=True)
def _(mo):
    batch_manual_btn = mo.ui.run_button(label="▶ Обработать ВСЕ (batch)")
    mo.vstack([
        mo.md("Lanczos+ESRGAN на **всех** загруженных, сводка в таблице:"),
        batch_manual_btn,
    ])
    return (batch_manual_btn,)


@app.cell(hide_code=True)
def _(
    batch_manual_btn,
    cv2,
    decode_qr,
    esrgan_up,
    manual_images,
    mo,
    wechat_det,
):
    def _batch_process(images: list, esrgan: object, wechat: object, triggered: bool) -> object:
        if not triggered:
            return mo.md("_Нажми ▶ Обработать ВСЕ_")
        if not images:
            return mo.callout(mo.md("Загрузи изображения сначала"), kind="warn")
        import pandas as _pd
        rows = []
        for m in images:
            img = m["img"]
            raw_dec = decode_qr(img, wechat)
            lan = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)
            lan_dec = decode_qr(lan, wechat)
            esr_dec = []
            if esrgan is not None:
                try:
                    up, _ = esrgan.enhance(img, outscale=4)
                    esr_dec = decode_qr(up, wechat)
                except Exception:
                    pass
            any_ok = bool(raw_dec or lan_dec or esr_dec)
            rows.append({
                "файл": m["name"],
                "размер": f"{img.shape[1]}×{img.shape[0]}",
                "raw": " | ".join(raw_dec) or "—",
                "lanczos×4": " | ".join(lan_dec) or "—",
                "esrgan×4": " | ".join(esr_dec) or "—",
                "✓": "✅" if any_ok else "❌",
            })
        df = _pd.DataFrame(rows)
        n_ok = sum(1 for r in rows if r["✓"] == "✅")
        return mo.vstack([
            mo.callout(mo.md(f"Декодировано: **{n_ok}/{len(rows)}**"),
                       kind="success" if n_ok else "warn"),
            mo.ui.table(df, selection=None),
        ])

    _batch_process(manual_images, esrgan_up, wechat_det, batch_manual_btn.value)
    return


@app.cell
def _(cv2, esrgan_up, frame_data, full_esrgan_btn, mo, plt):
    def _show_full(data: dict | None, esrgan: object, triggered: bool) -> object:
        if not triggered:
            return mo.md("_Нажми ▶ ESRGAN на полный кроп_")
        if data is None or data["crop"] is None:
            return mo.callout(mo.md("Загрузи кадр сначала"), kind="warn")

        crop = data["crop"]
        h, w = crop.shape[:2]

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Сырой кроп  {w}×{h}px", fontsize=12)
        axes[0].axis("off")

        if esrgan is not None:
            try:
                up, _ = esrgan.enhance(crop, outscale=4)
                axes[1].imshow(cv2.cvtColor(up, cv2.COLOR_BGR2RGB))
                axes[1].set_title(
                    f"ESRGAN ×4 → {up.shape[1]}×{up.shape[0]}px\n(используй для OCR)",
                    fontsize=12,
                )
            except Exception as e:
                axes[1].text(0.5, 0.5, f"Ошибка: {e}", ha="center", transform=axes[1].transAxes)
        else:
            up_l = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)
            axes[1].imshow(cv2.cvtColor(up_l, cv2.COLOR_BGR2RGB))
            axes[1].set_title(
                f"Lanczos ×4 → {up_l.shape[1]}×{up_l.shape[0]}px\n(ESRGAN не загружен)",
                fontsize=12,
            )
        axes[1].axis("off")
        plt.tight_layout()
        return fig

    _show_full(frame_data, esrgan_up, full_esrgan_btn.value)
    return


if __name__ == "__main__":
    app.run()
