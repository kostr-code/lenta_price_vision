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
    import zxingcpp
    from pyzbar.pyzbar import decode as pyzbar_decode

    return cv2, mo, mpatches, pd, plt, pyzbar_decode, zxingcpp


@app.cell
def _(mo):
    DATASETS = {
        "43_15 (стоп-кадры, 29 ценников)": {
            "csv": "Данные/43_15/43_15.csv",
            "video": "Данные/43_15/43_15.mp4",
        },
        "25_12-20 (движение, 57 ценников)": {
            "csv": "Данные/25_12-20/25_12-20.csv",
            "video": "Данные/25_12-20/25_12-20.mp4",
        },
    }
    dataset_picker = mo.ui.dropdown(
        list(DATASETS.keys()),
        value="43_15 (стоп-кадры, 29 ценников)",
        label="Датасет",
    )
    return DATASETS, dataset_picker


@app.cell
def _(DATASETS, dataset_picker, pd):
    _cfg = DATASETS[dataset_picker.value]
    _df = pd.read_csv(_cfg["csv"], dtype=str)
    _df.columns = _df.columns.str.strip()
    _df
    return


@app.cell
def _(DATASETS, dataset_picker, pd):
    _cfg = DATASETS[dataset_picker.value]
    _df = pd.read_csv(_cfg["csv"], dtype=str)
    _df.columns = _df.columns.str.strip()
    for _col in ["x_min", "y_min", "x_max", "y_max"]:
        _df[_col] = _df[_col].str.replace(",", ".").astype(float)
    _df["frame_timestamp"] = pd.to_numeric(_df["frame_timestamp"], errors="coerce")
    cfg = _cfg
    df = _df
    return cfg, df


@app.cell
def _(df):
    df
    return


@app.cell
def _(df, mo):
    tag_slider = mo.ui.slider(0, len(df) - 1, value=5, label="Ценник №")
    return (tag_slider,)


@app.cell
def _(tag_slider):
    tag_slider
    return


@app.cell
def _(df, tag_slider):
    row = df.iloc[tag_slider.value]
    row
    return (row,)


@app.cell
def _(dataset_picker, mo, row, tag_slider):
    mo.vstack(
        [
            mo.hstack([dataset_picker, tag_slider]),
            mo.md(
                f"**{row.get('product_name', '')}**  \n"
                f"`ts={int(row['frame_timestamp'])}ms` | "
                f"bbox (ориг.): x=[{row['x_min']:.0f}…{row['x_max']:.0f}]  "
                f"y=[{row['y_min']:.0f}…{row['y_max']:.0f}]  \n"
                f"color=`{row.get('color', '')}` | "
                f"qr_barcode=`{row.get('qr_code_barcode', '—')}`"
            ),
        ]
    )
    return


@app.cell
def _(cfg, cv2, row):
    _cap = cv2.VideoCapture(cfg["video"])
    _cap.set(cv2.CAP_PROP_POS_MSEC, float(row["frame_timestamp"]))
    _ok, _frame = _cap.read()
    _cap.release()
    raw_frame = _frame if _ok else None
    raw_frame
    return (raw_frame,)


@app.cell
def _(cv2, mpatches, plt, raw_frame, row):
    def show_original_with_bbox(frame, row):
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.add_patch(
            mpatches.Rectangle(
                (row["x_min"], row["y_min"]),
                row["x_max"] - row["x_min"],
                row["y_max"] - row["y_min"],
                linewidth=3,
                edgecolor="lime",
                facecolor="none",
            )
        )
        ax.set_title(
            f"Шаг 1 — оригинальный кадр {frame.shape[1]}×{frame.shape[0]}px  |  "
            f"bbox из CSV: x=[{row['x_min']:.0f}…{row['x_max']:.0f}]  "
            f"y=[{row['y_min']:.0f}…{row['y_max']:.0f}]",
            fontsize=11,
        )
        ax.axis("off")
        plt.tight_layout()
        return fig


    show_original_with_bbox(raw_frame, row) if raw_frame is not None else None
    return


@app.cell
def _(cv2, mpatches, plt, raw_frame, row):
    def show_cw_vs_ccw(frame, row):
        _H, _W = frame.shape[:2]  # 2160, 3840

        # CW: new_x = H−1−y,  new_y = x_orig
        rot_cw = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        cw_x1 = _H - 1 - row["y_max"]
        cw_y1 = row["x_min"]
        cw_w = row["y_max"] - row["y_min"]
        cw_h = row["x_max"] - row["x_min"]

        # CCW: new_x = y_orig,  new_y = W−1−x_orig
        rot_ccw = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ccw_x1 = row["y_min"]
        ccw_y1 = _W - 1 - row["x_max"]
        ccw_w = row["y_max"] - row["y_min"]
        ccw_h = row["x_max"] - row["x_min"]

        fig, (ax_cw, ax_ccw) = plt.subplots(1, 2, figsize=(16, 9))

        ax_cw.imshow(cv2.cvtColor(rot_cw, cv2.COLOR_BGR2RGB))
        ax_cw.add_patch(
            mpatches.Rectangle(
                (cw_x1, cw_y1),
                cw_w,
                cw_h,
                linewidth=3,
                edgecolor="lime",
                facecolor="none",
            )
        )
        ax_cw.set_title("↻ CW (по часовой)", fontsize=13)
        ax_cw.axis("off")

        ax_ccw.imshow(cv2.cvtColor(rot_ccw, cv2.COLOR_BGR2RGB))
        ax_ccw.add_patch(
            mpatches.Rectangle(
                (ccw_x1, ccw_y1),
                ccw_w,
                ccw_h,
                linewidth=3,
                edgecolor="lime",
                facecolor="none",
            )
        )
        ax_ccw.set_title("↺ CCW (против часовой)", fontsize=13)
        ax_ccw.axis("off")

        fig.suptitle("Шаг 2 — какой bbox попадает на ценник?", fontsize=14)
        plt.tight_layout()
        return fig


    show_cw_vs_ccw(raw_frame, row) if raw_frame is not None else None
    return


@app.cell
def _(mo):
    rotation_radio = mo.ui.radio(
        options={"↺ CCW": "CCW", "↻ CW": "CW"},
        value="↺ CCW",
        label="Правильный поворот",
    )
    padding_slider = mo.ui.slider(0, 300, value=30, step=10, label="Padding (px)")
    return padding_slider, rotation_radio


@app.cell
def _(mo, padding_slider, rotation_radio):
    mo.hstack([rotation_radio, padding_slider])
    return


@app.cell
def _(cv2, padding_slider, raw_frame, rotation_radio, row):
    def make_crop(frame, row, rotation, pad):
        _H, _W = frame.shape[:2]
        if rotation == "CCW":
            rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            bx1 = int(row["y_min"])
            by1 = int(_W - 1 - row["x_max"])
            bx2 = int(row["y_max"])
            by2 = int(_W - 1 - row["x_min"])
        else:
            rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            bx1 = int(_H - 1 - row["y_max"])
            by1 = int(row["x_min"])
            bx2 = int(_H - 1 - row["y_min"])
            by2 = int(row["x_max"])
        fh, fw = rotated.shape[:2]
        return rotated[
            max(0, by1 - pad) : min(fh, by2 + pad),
            max(0, bx1 - pad) : min(fw, bx2 + pad),
        ]


    crop = (
        make_crop(raw_frame, row, rotation_radio.value, padding_slider.value)
        if raw_frame is not None
        else None
    )
    return (crop,)


@app.cell
def _(crop, cv2, plt):
    def show_crop(crop):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title(
            f"Шаг 3 — кроп {crop.shape[1]}×{crop.shape[0]}px", fontsize=12
        )
        ax.axis("off")
        plt.tight_layout()
        return fig


    show_crop(crop) if (crop is not None and crop.size > 0) else None
    return


@app.cell
def _(crop, cv2, mo, pyzbar_decode, zxingcpp):
    def try_decode(crop):
        _gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        found = []
        for scale in [1, 2, 3, 4]:
            img = cv2.resize(
                _gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4
            )
            for r in pyzbar_decode(img):
                found.append(
                    f"✅ **pyzbar ×{scale}**: `{r.type}` → `{r.data.decode()}`"
                )
            for r in zxingcpp.read_barcodes(img):
                found.append(f"✅ **zxing ×{scale}**: `{r.format}` → `{r.text}`")
        return found


    if crop is not None and crop.size > 0:
        _results = try_decode(crop)
        if _results:
            mo.callout(
                mo.md("**Шаг 4 — декодировано:**\n\n" + "\n\n".join(_results)),
                kind="success",
            )
        else:
            mo.callout(
                mo.md(
                    "**Шаг 4 — не декодировано.** Увеличь Padding или смени ценник."
                ),
                kind="warn",
            )
    return (try_decode,)


@app.cell
def _(crop, try_decode):
    try_decode(crop)
    return


@app.cell
def _(crop, cv2, mo):
    def wechat_decode(img_bgr):
        _det = cv2.wechat_qrcode_WeChatQRCode()
        found = []
        for scale in [1, 2, 4]:
            _img = cv2.resize(img_bgr, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_LANCZOS4)
            _data, _pts = _det.detectAndDecode(_img)
            for d in _data:
                if d:
                    found.append(f"✅ WeChat ×{scale}: `{d}`")
        return found

    def wechat_widget(crop):
        if crop is None or crop.size == 0:
            return mo.md("_Нет кропа_")
        results = wechat_decode(crop)
        if results:
            return mo.callout(
                mo.md("**Шаг 4б — WeChat QR декодировано:**\n\n" + "\n\n".join(results)),
                kind="success",
            )
        return mo.callout(mo.md("**Шаг 4б — WeChat QR: не декодирован**"), kind="warn")

    wechat_widget(crop)
    return


@app.cell
def _(cv2, mo, pyzbar_decode, raw_frame, rotation_radio, zxingcpp):
    def decode_full_frame(frame, rotation):
        rotated = (
            cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if rotation == "CCW"
            else cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        )
        _gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        _wechat = cv2.wechat_qrcode_WeChatQRCode()
        found = []
        for r in pyzbar_decode(_gray):
            found.append(f"✅ pyzbar: {r.type} → {r.data.decode()}")
        for r in zxingcpp.read_barcodes(_gray):
            found.append(f"✅ zxing: {r.format} → {r.text}")
        _data, _ = _wechat.detectAndDecode(rotated)
        for d in _data:
            if d:
                found.append(f"✅ WeChat: {d}")
        return found

    def fullframe_widget(frame, rotation):
        if frame is None:
            return mo.md("_Нет кадра_")
        results = decode_full_frame(frame, rotation)
        if results:
            return mo.callout(
                mo.md("**Шаг 4в — QR на полном кадре:**\n\n" + "\n\n".join(results)),
                kind="success",
            )
        return mo.callout(
            mo.md("**Шаг 4в — QR на полном кадре не найден**"),
            kind="warn",
        )

    fullframe_widget(raw_frame, rotation_radio.value)
    return


@app.cell
def _(mo):
    scan_window = mo.ui.slider(
        5, 300,
        value=30,
        step=1,
        label="Окно (кадров в каждую сторону)"
    )
    scan_button = mo.ui.run_button(label="▶ Запустить сканирование резкости")
    return scan_button, scan_window


@app.cell
def _(mo, scan_button, scan_window):
    mo.vstack([
        mo.md("## Шаг 5 — поиск самого резкого кадра"),
        mo.md(
            "Сканируем +-N кадров вокруг CSV timestamp.  \n"
            "`Laplacian variance` — мера резкости: чем выше, тем чётче изображение.  \n"
            "🔴 красная линия = CSV timestamp, 🟢 зелёная = лучший найденный кадр."
        ),
        mo.hstack([scan_window, scan_button]),
    ])
    return


@app.cell
def _(cfg, cv2, padding_slider, rotation_radio, row, scan_button, scan_window):
    def sharpness_scan(video_path, ts_ms, row, rotation, pad, n_frames):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_ms = 1000.0 / fps
        W_ORIG = 3840

        timestamps, variances = [], []
        best_var, best_crop, best_ts = -1.0, None, ts_ms

        for i in range(-n_frames, n_frames + 1):
            ts = ts_ms + i * frame_ms
            cap.set(cv2.CAP_PROP_POS_MSEC, ts)
            ok, frame = cap.read()
            if not ok:
                continue

            if rotation == "CCW":
                rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                bx1 = int(row["y_min"])
                by1 = int(W_ORIG - 1 - row["x_max"])
                bx2 = int(row["y_max"])
                by2 = int(W_ORIG - 1 - row["x_min"])
            else:
                H_ORIG = 2160
                rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                bx1 = int(H_ORIG - 1 - row["y_max"])
                by1 = int(row["x_min"])
                bx2 = int(H_ORIG - 1 - row["y_min"])
                by2 = int(row["x_max"])

            fh, fw = rotated.shape[:2]
            crop = rotated[
                max(0, by1 - pad) : min(fh, by2 + pad),
                max(0, bx1 - pad) : min(fw, bx2 + pad),
            ]
            if crop.size == 0:
                continue

            lap_var = cv2.Laplacian(
                cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F
            ).var()
            timestamps.append(ts)
            variances.append(lap_var)

            if lap_var > best_var:
                best_var, best_crop, best_ts = lap_var, crop.copy(), ts

        cap.release()
        return timestamps, variances, best_crop, best_ts, best_var


    if scan_button.value:
        scan_ts, scan_var, best_crop, best_ts, best_var = sharpness_scan(
            cfg["video"],
            float(row["frame_timestamp"]),
            row,
            rotation_radio.value,
            padding_slider.value,
            scan_window.value,
        )
    else:
        scan_ts = scan_var = best_crop = best_ts = best_var = None
    return best_crop, best_ts, best_var, scan_ts, scan_var


@app.cell
def _(best_crop, best_ts, best_var, cv2, mo, plt, row, scan_ts, scan_var):
    def plot_scan(ts_list, var_list, b_ts, b_var, b_crop, csv_ts):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        ax1.plot(ts_list, var_list, "b-o", markersize=4, linewidth=1.5)
        ax1.axvline(
            csv_ts, color="red", lw=2, ls="--", label=f"CSV ts = {csv_ts:.0f} ms"
        )
        ax1.axvline(
            b_ts,
            color="lime",
            lw=2,
            ls="-",
            label=f"Лучший ts = {b_ts:.0f} ms  (Δ={b_ts - csv_ts:+.0f} ms)",
        )
        ax1.set_xlabel("Timestamp (ms)")
        ax1.set_ylabel("Laplacian variance")
        ax1.set_title("Резкость кропа по кадрам")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        if b_crop is not None:
            ax2.imshow(cv2.cvtColor(b_crop, cv2.COLOR_BGR2RGB))
            ax2.set_title(
                f"Лучший кроп  ts={b_ts:.0f}ms  var={b_var:.1f}", fontsize=11
            )
        ax2.axis("off")
        plt.tight_layout()
        return fig


    _run_scan = lambda: plot_scan(
        scan_ts,
        scan_var,
        best_ts,
        best_var,
        best_crop,
        float(row["frame_timestamp"]),
    )

    _run_scan() if scan_ts is not None else mo.md(
        "Нажми **▶ Запустить** выше чтобы начать сканирование."
    )
    return


@app.cell
def _(best_crop, cv2, mo, pyzbar_decode, zxingcpp):
    def try_decode_best(crop):
        _gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _wechat = cv2.wechat_qrcode_WeChatQRCode()
        found = []
        for scale in [1, 2, 3, 4]:
            _img_g = cv2.resize(
                _gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4
            )
            _img_b = cv2.resize(
                crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4
            )
            for r in pyzbar_decode(_img_g):
                found.append(f"✅ pyzbar ×{scale}: {r.type} → {r.data.decode()}")
            for r in zxingcpp.read_barcodes(_img_g):
                found.append(f"✅ zxing ×{scale}: {r.format} → {r.text}")
            _data, _ = _wechat.detectAndDecode(_img_b)
            for d in _data:
                if d:
                    found.append(f"✅ WeChat ×{scale}: {d}")
        return found


    def decode_widget(crop):
        if crop is None or crop.size == 0:
            return mo.md("_Запусти сканирование резкости выше._")
        results = try_decode_best(crop)
        if results:
            return mo.callout(
                mo.md(
                    "**QR/штрихкод на лучшем кадре:**\n\n" + "\n\n".join(results)
                ),
                kind="success",
            )
        return mo.callout(
            mo.md(
                "**QR на лучшем кадре не декодирован.**  \n"
                "Попробуй увеличить Padding или окно сканирования."
            ),
            kind="warn",
        )


    decode_widget(best_crop)
    return


if __name__ == "__main__":
    app.run()
