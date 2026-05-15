"""
crop_explorer.py — Gradio UI для визуального тестирования scoring.

Читает кэш созданный track_and_crop.py (pass1) и позволяет:
- Листать треки (◄/►)
- Видеть всех кандидатов трека, отсортированных по текущей формуле скоринга
- Менять веса scoring через слайдеры и мгновенно видеть нового победителя
- Сравнивать: winner(текущие веса) vs winner(только Laplacian)

Запуск:
    uv run python packages/ml/crop_explorer.py --cache runs/crops/26_2-10/cache
    uv run python packages/ml/crop_explorer.py  # выбрать кэш через UI
"""

import argparse
import json
from pathlib import Path

import gradio as gr

# ── scoring (дублируем из track_and_crop чтобы не тянуть весь модуль) ────────


def compute_score(
    c: dict,
    w_sharp: float,
    w_conf: float,
    w_flow: float,
    w_age: float,
) -> float:
    motion_ok = 1.0 / (1.0 + c.get("flow_mag", 0.0) * 0.5)
    sharpness = c.get("tenengrad", 0.0) / 1000
    age_bonus = min(c.get("track_age", 0) / 5, 2.0)
    return (
        w_sharp * sharpness
        + w_conf * c.get("conf", 0.0) ** 2 * 100
        + w_flow * motion_ok * 100
        + w_age * age_bonus * 100
    )


def laplacian_score(c: dict) -> float:
    return c.get("laplacian", 0.0)


# ── кэш ──────────────────────────────────────────────────────────────────────


def load_cache(cache_dir: str) -> dict | None:
    p = Path(cache_dir) / "tracks.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def track_ids_sorted(cache: dict) -> list[str]:
    """Сортируем треки по track_id."""
    return sorted(cache["tracks"].keys(), key=lambda k: int(k))


# ── gallery helpers ───────────────────────────────────────────────────────────


def build_gallery(
    track: dict,
    w_sharp: float,
    w_conf: float,
    w_flow: float,
    w_age: float,
) -> list[tuple[str, str]]:
    cands = track["candidates"]
    scored = sorted(
        cands,
        key=lambda c: compute_score(c, w_sharp, w_conf, w_flow, w_age),
        reverse=True,
    )
    result = []
    for c in scored:
        sc = compute_score(c, w_sharp, w_conf, w_flow, w_age)
        lap = laplacian_score(c)
        label = (
            f"sc={sc:.0f}  fl={c.get('flow_mag', 0):.1f}  "
            f"cf={c.get('conf', 0):.2f}  lap={lap:.0f}  "
            f"fr={c.get('frame_idx', 0)}"
        )
        path = c.get("crop_path", "")
        if Path(path).exists():
            result.append((path, label))
    return result


def winner_info(
    track: dict,
    w_sharp: float,
    w_conf: float,
    w_flow: float,
    w_age: float,
    use_laplacian: bool = False,
) -> tuple[str | None, str]:
    cands = track["candidates"]
    if not cands:
        return None, "нет кандидатов"

    if use_laplacian:
        best = max(cands, key=laplacian_score)
        sc = laplacian_score(best)
        method = "Laplacian"
    else:
        best = max(cands, key=lambda c: compute_score(c, w_sharp, w_conf, w_flow, w_age))
        sc = compute_score(best, w_sharp, w_conf, w_flow, w_age)
        method = "Composite"

    info = (
        f"**{method}** | score={sc:.1f}\n\n"
        f"frame={best.get('frame_idx')}  "
        f"conf={best.get('conf', 0):.2f}  "
        f"flow={best.get('flow_mag', 0):.2f}  "
        f"age={best.get('track_age', 0)}  "
        f"lap={best.get('laplacian', 0):.0f}  "
        f"ten={best.get('tenengrad', 0):.0f}"
    )
    path = best.get("crop_path", "")
    return path if Path(path).exists() else None, info


def track_stats(track: dict) -> str:
    cands = track["candidates"]
    if not cands:
        return ""
    confs = [c.get("conf", 0) for c in cands]
    flows = [c.get("flow_mag", 0) for c in cands]
    return (
        f"Кандидатов: {len(cands)} | "
        f"conf avg={sum(confs) / len(confs):.2f} max={max(confs):.2f} | "
        f"flow avg={sum(flows) / len(flows):.2f} min={min(flows):.2f}"
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────


def build_ui(initial_cache_dir: str = "") -> gr.Blocks:
    with gr.Blocks(title="Crop Explorer", theme=gr.themes.Soft()) as demo:
        # state
        cache_state = gr.State(None)
        track_ids_state = gr.State([])
        track_idx_state = gr.State(0)

        gr.Markdown("# Crop Explorer — визуальный тест scoring")

        # ── блок загрузки ──
        with gr.Row():
            cache_input = gr.Textbox(
                label="Путь к cache/ директории",
                value=initial_cache_dir,
                placeholder="/path/to/runs/crops/26_2-10/cache",
                scale=4,
            )
            load_btn = gr.Button("Загрузить", variant="primary", scale=1)
        load_status = gr.Markdown("_Укажи путь и нажми Загрузить_")

        # ── навигация по трекам ──
        with gr.Row():
            prev_btn = gr.Button("◄ пред", scale=1)
            track_label = gr.Markdown("**Трек —**")
            next_btn = gr.Button("след ►", scale=1)
        stats_label = gr.Markdown("")

        # ── слайдеры + победители ──
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Веса скоринга")
                w_sharp = gr.Slider(0, 1, value=0.3, step=0.05, label="Tenengrad (резкость)")
                w_conf = gr.Slider(0, 1, value=0.3, step=0.05, label="YOLO confidence")
                w_flow = gr.Slider(0, 1, value=0.3, step=0.05, label="Optical flow (0=стоит)")
                w_age = gr.Slider(0, 1, value=0.1, step=0.05, label="Track age (длина трека)")

            with gr.Column(scale=1):
                gr.Markdown("### Победитель (composite)")
                winner_img = gr.Image(label="", show_label=False, height=250)
                winner_info_md = gr.Markdown("")

            with gr.Column(scale=1):
                gr.Markdown("### Победитель (только Laplacian)")
                lap_winner_img = gr.Image(label="", show_label=False, height=250)
                lap_winner_info_md = gr.Markdown("")

        # ── галерея кандидатов ──
        gr.Markdown("### Все кандидаты трека (по score ↓)")
        gallery = gr.Gallery(
            label="",
            show_label=False,
            columns=8,
            height=220,
            object_fit="contain",
        )

        # ── helpers ──────────────────────────────────────────────────────────

        def do_load(cache_dir: str):
            cache = load_cache(cache_dir.strip())
            if cache is None:
                return (
                    None,
                    [],
                    0,
                    "❌ tracks.json не найден",
                    "**Трек —**",
                    "",
                    [],
                    None,
                    "",
                    None,
                    "",
                )
            ids = track_ids_sorted(cache)
            n = len(ids)
            status = f"✅ Загружено: **{n} треков** из `{cache_dir}`"
            return cache, ids, 0, status, *render_track(cache, ids, 0, 0.3, 0.3, 0.3, 0.1)

        def render_track(cache, ids, idx, ws, wc, wf, wa):
            if not ids or cache is None:
                return "**Трек —**", "", [], None, "", None, ""
            tid = ids[idx]
            track = cache["tracks"][tid]
            label = f"**Трек #{tid}** ({idx + 1}/{len(ids)})"
            stats = track_stats(track)
            gallery_items = build_gallery(track, ws, wc, wf, wa)
            w_path, w_info = winner_info(track, ws, wc, wf, wa, use_laplacian=False)
            l_path, l_info = winner_info(track, ws, wc, wf, wa, use_laplacian=True)
            return label, stats, gallery_items, w_path, w_info, l_path, l_info

        def go_prev(cache, ids, idx, ws, wc, wf, wa):
            idx = max(0, idx - 1)
            return idx, *render_track(cache, ids, idx, ws, wc, wf, wa)

        def go_next(cache, ids, idx, ws, wc, wf, wa):
            idx = min(len(ids) - 1, idx + 1) if ids else 0
            return idx, *render_track(cache, ids, idx, ws, wc, wf, wa)

        def on_sliders(cache, ids, idx, ws, wc, wf, wa):
            return render_track(cache, ids, idx, ws, wc, wf, wa)

        render_outputs = [
            track_label,
            stats_label,
            gallery,
            winner_img,
            winner_info_md,
            lap_winner_img,
            lap_winner_info_md,
        ]

        load_btn.click(
            do_load,
            inputs=[cache_input],
            outputs=[
                cache_state,
                track_ids_state,
                track_idx_state,
                load_status,
                *render_outputs,
            ],
        )

        prev_btn.click(
            go_prev,
            inputs=[cache_state, track_ids_state, track_idx_state, w_sharp, w_conf, w_flow, w_age],
            outputs=[track_idx_state, *render_outputs],
        )
        next_btn.click(
            go_next,
            inputs=[cache_state, track_ids_state, track_idx_state, w_sharp, w_conf, w_flow, w_age],
            outputs=[track_idx_state, *render_outputs],
        )

        for slider in [w_sharp, w_conf, w_flow, w_age]:
            slider.change(
                on_sliders,
                inputs=[
                    cache_state,
                    track_ids_state,
                    track_idx_state,
                    w_sharp,
                    w_conf,
                    w_flow,
                    w_age,
                ],
                outputs=render_outputs,
            )

    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="", help="Путь к cache/ директории")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    demo = build_ui(initial_cache_dir=args.cache)
    demo.launch(server_name=args.host, server_port=args.port, share=False)
