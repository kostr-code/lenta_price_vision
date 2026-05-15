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
    import pathlib
    import zxingcpp
    from pyzbar.pyzbar import decode as pyzbar_decode

    return cv2, mo, np, pathlib, pd, plt, pyzbar_decode, zxingcpp


@app.cell
def _(cv2, mo, pathlib):
    import urllib.request as _urllib_req
    _MODEL_DIR = pathlib.Path("wechat_models")
    _FILES = ["detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel"]
    _BASE = "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/"
    _MODEL_DIR.mkdir(exist_ok=True)
    for _f in _FILES:
        if not (_MODEL_DIR / _f).exists():
            _urllib_req.urlretrieve(_BASE + _f, _MODEL_DIR / _f)
    try:
        wechat_detector = cv2.wechat_qrcode_WeChatQRCode(
            str(_MODEL_DIR / "detect.prototxt"),
            str(_MODEL_DIR / "detect.caffemodel"),
            str(_MODEL_DIR / "sr.prototxt"),
            str(_MODEL_DIR / "sr.caffemodel"),
        )
        _msg = mo.callout(mo.md("**WeChat: NN-модели загружены** (детекция + SR)"), kind="success")
    except Exception as _e:
        wechat_detector = cv2.wechat_qrcode_WeChatQRCode()
        _msg = mo.callout(mo.md(f"**WeChat: базовый режим** (без NN). Ошибка: `{_e}`"), kind="warn")
    _msg
    return (wechat_detector,)


@app.cell
def _(mo):
    try:
        from qreader import QReader as _QReader
        qr_reader_inst = _QReader()
        _msg = mo.callout(mo.md("**qreader: YOLOv8-детектор загружен**"), kind="success")
    except Exception as _e:
        qr_reader_inst = None
        _msg = mo.callout(mo.md(f"**qreader: недоступен** — `{_e}`"), kind="warn")
    _msg
    return (qr_reader_inst,)


@app.cell
def _(cv2, pyzbar_decode, qr_reader_inst, wechat_detector, zxingcpp):
    def preprocess_variants(gray):
        """Несколько вариантов предобработки grayscale изображения."""
        _variants = [("raw", gray)]
        _clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        _variants.append(("CLAHE", _clahe.apply(gray)))
        _blur = cv2.GaussianBlur(gray, (0, 0), 3)
        _variants.append(("unsharp", cv2.addWeighted(gray, 1.5, _blur, -0.5, 0)))
        _variants.append(("adapt_thresh", cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )))
        return _variants

    def decode_all(img_bgr):
        """Все декодеры + preprocessing на одном BGR изображении.
        Возвращает list of (decoder, scale, preprocessing, value).
        """
        _gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        found = []
        seen = set()

        # pyzbar + zxing: все preprocessing × масштабы
        for _prep_name, _prep_gray in preprocess_variants(_gray):
            for _scale in [1, 2, 4]:
                _g = cv2.resize(_prep_gray, None, fx=_scale, fy=_scale,
                                interpolation=cv2.INTER_LANCZOS4)
                for r in pyzbar_decode(_g):
                    _val = r.data.decode()
                    if _val and ("pyzbar", _val) not in seen:
                        seen.add(("pyzbar", _val))
                        found.append(("pyzbar", _scale, _prep_name, _val))
                for r in zxingcpp.read_barcodes(_g):
                    if r.text and ("zxing", r.text) not in seen:
                        seen.add(("zxing", r.text))
                        found.append(("zxing", _scale, _prep_name, r.text))

        # WeChat с NN-моделями: raw BGR × масштабы
        for _scale in [1, 2, 4]:
            _b = cv2.resize(img_bgr, None, fx=_scale, fy=_scale,
                            interpolation=cv2.INTER_LANCZOS4)
            _data, _ = wechat_detector.detectAndDecode(_b)
            for d in _data:
                if d and ("WeChat", d) not in seen:
                    seen.add(("WeChat", d))
                    found.append(("WeChat", _scale, "raw", d))

        # qreader (YOLOv8): один раз на оригинале
        if qr_reader_inst is not None:
            print(f"=> qreader: ${qr_reader_inst}")
            try:
                _rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                for v in (qr_reader_inst.detect_and_decode(image=_rgb) or []):
                    if v and ("qreader", v) not in seen:
                        seen.add(("qreader", v))
                        found.append(("qreader", 1, "raw", v))
            except Exception:
                pass

        return found

    return (decode_all,)


@app.cell
def _(mo):
    mo.md("""
    ## Секция 1 — Загрузи изображение: битва декодеров
    """)
    return


@app.cell
def _(mo):
    upload = mo.ui.file(filetypes=[".jpg", ".jpeg", ".png"], label="Загрузи изображение с QR")
    return (upload,)


@app.cell
def _(upload):
    upload
    return


@app.cell
def _(cv2, decode_all, mo, np, pd, plt, upload):
    def upload_widget(upload_val):
        if not upload_val:
            return mo.md("_Загрузи изображение выше_")
        _buf = np.frombuffer(upload_val[0].contents, np.uint8)
        _img = cv2.imdecode(_buf, cv2.IMREAD_COLOR)
        if _img is None:
            return mo.callout(mo.md("**Не удалось прочитать изображение**"), kind="danger")
        _fig, _ax = plt.subplots(figsize=(10, 6))
        _ax.imshow(cv2.cvtColor(_img, cv2.COLOR_BGR2RGB))
        _ax.set_title(f"{upload_val[0].name}  |  {_img.shape[1]}×{_img.shape[0]}px", fontsize=11)
        _ax.axis("off")
        plt.tight_layout()
        _results = decode_all(_img)
        if not _results:
            return mo.vstack([
                _fig,
                mo.callout(mo.md("**Ни один декодер не нашёл QR/штрихкод**"), kind="warn"),
            ])
        _df = pd.DataFrame(_results, columns=["Декодер", "Масштаб", "Препроц.", "Значение"])
        return mo.vstack([
            _fig,
            mo.callout(mo.md(f"**Найдено результатов: {len(_results)}**"), kind="success"),
            mo.ui.table(_df),
        ])

    upload_widget(upload.value)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Секция 2 — Нарезаем лучшие кропы из размеченных видео
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
    }
    harvest_dataset = mo.ui.dropdown(
        list(DATASETS.keys()), value="43_15", label="Датасет"
    )
    harvest_window = mo.ui.slider(
        5, 60, value=20, step=5, label="±N кадров для поиска резкости"
    )
    harvest_button = mo.ui.run_button(label="▶ Нарезать кропы")
    return DATASETS, harvest_button, harvest_dataset, harvest_window


@app.cell
def _(harvest_button, harvest_dataset, harvest_window, mo):
    mo.vstack([
        mo.md(
            "Для каждого ценника: найти самый резкий кадр (Laplacian variance) "
            "в окне ±N вокруг CSV timestamp и сохранить кроп в `qr_crops/{датасет}/`."
        ),
        mo.hstack([harvest_dataset, harvest_window, harvest_button]),
    ])
    return


@app.cell
def _(
    DATASETS,
    cv2,
    harvest_button,
    harvest_dataset,
    harvest_window,
    pathlib,
    pd,
):
    def _find_best_crop(video_path, ts_ms, row, n_frames):
        W_ORIG = 3840
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_ms = 1000.0 / fps
        best_var, best_c = -1.0, None
        for i in range(-n_frames, n_frames + 1):
            ts = ts_ms + i * frame_ms
            cap.set(cv2.CAP_PROP_POS_MSEC, ts)
            ok, frame = cap.read()
            if not ok:
                continue
            rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            bx1 = int(row["y_min"])
            by1 = int(W_ORIG - 1 - row["x_max"])
            bx2 = int(row["y_max"])
            by2 = int(W_ORIG - 1 - row["x_min"])
            fh, fw = rotated.shape[:2]
            c = rotated[max(0, by1):min(fh, by2), max(0, bx1):min(fw, bx2)]
            if c.size == 0:
                continue
            lap = cv2.Laplacian(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if lap > best_var:
                best_var, best_c = lap, c.copy()
        cap.release()
        return best_c, best_var

    if harvest_button.value:
        _cfg = DATASETS[harvest_dataset.value]
        _df = pd.read_csv(_cfg["csv"], dtype=str)
        _df.columns = _df.columns.str.strip()
        for _col in ["x_min", "y_min", "x_max", "y_max"]:
            _df[_col] = _df[_col].str.replace(",", ".").astype(float)
        _df["frame_timestamp"] = pd.to_numeric(_df["frame_timestamp"], errors="coerce")

        _out = pathlib.Path("qr_crops") / harvest_dataset.value
        _out.mkdir(parents=True, exist_ok=True)

        _meta_rows, _saved_files = [], []
        for _idx, _row in _df.iterrows():
            _crop, _var = _find_best_crop(
                _cfg["video"], float(_row["frame_timestamp"]), _row, harvest_window.value
            )
            if _crop is not None and _crop.size > 0:
                _fname = f"{int(_idx):03d}.jpg"
                cv2.imwrite(str(_out / _fname), _crop)
                _saved_files.append(_fname)
                _meta_rows.append({
                    "filename": _fname,
                    "product_name": _row.get("product_name", ""),
                    "gt_qr": _row.get("qr_code_barcode", ""),
                    "lap_var": round(_var, 1),
                })
        pd.DataFrame(_meta_rows).to_csv(_out / "meta.csv", index=False)
        harvest_result = {"saved": _saved_files, "out_dir": str(_out)}
    else:
        harvest_result = None
    return (harvest_result,)


@app.cell
def _(cv2, harvest_result, mo, pathlib, plt):
    def harvest_widget(result):
        if result is None:
            return mo.md("_Нажми **▶ Нарезать кропы** выше_")
        _n = len(result["saved"])
        _badge = mo.callout(
            mo.md(f"**Сохранено {_n} кропов** → `{result['out_dir']}/`"),
            kind="success",
        )
        _paths = [pathlib.Path(result["out_dir"]) / f for f in result["saved"][:6]]
        _imgs = [cv2.imread(str(p)) for p in _paths if p.exists()]
        if not _imgs:
            return _badge
        _cols = len(_imgs)
        _fig, _axes = plt.subplots(1, _cols, figsize=(4 * _cols, 4))
        _axes = [_axes] if _cols == 1 else list(_axes)
        for _ax, _img in zip(_axes, _imgs):
            _ax.imshow(cv2.cvtColor(_img, cv2.COLOR_BGR2RGB))
            _ax.axis("off")
        _fig.suptitle(f"Превью первых {_cols} из {_n} кропов", fontsize=12)
        plt.tight_layout()
        return mo.vstack([_badge, _fig])

    harvest_widget(harvest_result)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Секция 3 — Батч-битва декодеров на сохранённых кропах
    """)
    return


@app.cell
def _(mo):
    BATCH_DATASETS = {"43_15": "qr_crops/43_15", "25_12-20": "qr_crops/25_12-20"}
    batch_dataset = mo.ui.dropdown(list(BATCH_DATASETS.keys()), value="43_15", label="Датасет")
    decode_button = mo.ui.run_button(label="▶ Запустить батч-декодирование")
    return BATCH_DATASETS, batch_dataset, decode_button


@app.cell
def _(BATCH_DATASETS, batch_dataset, decode_button, mo, pathlib):
    _crop_dir = pathlib.Path(BATCH_DATASETS[batch_dataset.value])
    _n = len(list(_crop_dir.glob("*.jpg"))) if _crop_dir.exists() else 0
    mo.vstack([
        mo.md(f"Кропов в `{_crop_dir}`: **{_n}**  _(нет кропов — сначала запусти Секцию 2)_"),
        mo.hstack([batch_dataset, decode_button]),
    ])
    return


@app.cell
def _(
    BATCH_DATASETS,
    batch_dataset,
    cv2,
    decode_all,
    decode_button,
    pathlib,
    pd,
):
    if decode_button.value:
        _crop_dir = pathlib.Path(BATCH_DATASETS[batch_dataset.value])
        _meta_path = _crop_dir / "meta.csv"
        _meta = pd.read_csv(_meta_path) if _meta_path.exists() else pd.DataFrame()

        _rows = []
        for _jpg in sorted(_crop_dir.glob("*.jpg")):
            _img = cv2.imread(str(_jpg))
            if _img is None:
                continue
            _hits = decode_all(_img)
            _pyzbar  = next((v for d, s, t, v in _hits if d == "pyzbar"), "")
            _zxing   = next((v for d, s, t, v in _hits if d == "zxing"), "")
            _wechat  = next((v for d, s, t, v in _hits if d == "WeChat"), "")
            _qreader = next((v for d, s, t, v in _hits if d == "qreader"), "")
            _any     = _pyzbar or _zxing or _wechat or _qreader
            _gt = ""
            if not _meta.empty and _jpg.name in _meta["filename"].values:
                _gt = str(_meta.loc[_meta["filename"] == _jpg.name, "gt_qr"].iloc[0])
            _match = bool(_any and _gt and _any.strip() == _gt.strip())
            _rows.append({
                "файл":            _jpg.name,
                "gt_qr":           _gt,
                "pyzbar":          _pyzbar,
                "zxing":           _zxing,
                "WeChat":          _wechat,
                "qreader":         _qreader,
                "совп. с GT":      "✅" if _match else ("—" if not _gt else "❌"),
            })
        batch_df = pd.DataFrame(_rows)
    else:
        batch_df = None
    return (batch_df,)


@app.cell
def _(batch_df, mo, pd):
    def batch_widget(df):
        if df is None:
            return mo.md("_Нажми **▶ Запустить батч-декодирование** выше_")
        _n = len(df)
        if _n == 0:
            return mo.callout(mo.md("**Кропов не найдено** — запусти Секцию 2"), kind="warn")
        _pyzbar_n  = int((df["pyzbar"] != "").sum())
        _zxing_n   = int((df["zxing"] != "").sum())
        _wechat_n  = int((df["WeChat"] != "").sum())
        _qreader_n = int((df["qreader"] != "").sum())
        _any_n     = int((
            (df["pyzbar"] != "") | (df["zxing"] != "") |
            (df["WeChat"] != "") | (df["qreader"] != "")
        ).sum())
        _match_n   = int((df["совп. с GT"] == "✅").sum())
        _stats = pd.DataFrame([
            {"Декодер": "pyzbar",    "Найдено": _pyzbar_n,  "% кропов": f"{100*_pyzbar_n//_n}%"},
            {"Декодер": "zxing",     "Найдено": _zxing_n,   "% кропов": f"{100*_zxing_n//_n}%"},
            {"Декодер": "WeChat+NN", "Найдено": _wechat_n,  "% кропов": f"{100*_wechat_n//_n}%"},
            {"Декодер": "qreader",   "Найдено": _qreader_n, "% кропов": f"{100*_qreader_n//_n}%"},
            {"Декодер": "Хоть кто", "Найдено": _any_n,     "% кропов": f"{100*_any_n//_n}%"},
            {"Декодер": "== GT",     "Найдено": _match_n,   "% кропов": f"{100*_match_n//_n}%"},
        ])
        return mo.vstack([
            mo.md(f"### Итого кропов: {_n}"),
            mo.ui.table(_stats),
            mo.md("### Детализация по кропам"),
            mo.ui.table(df),
        ])

    batch_widget(batch_df)
    return


if __name__ == "__main__":
    app.run()
