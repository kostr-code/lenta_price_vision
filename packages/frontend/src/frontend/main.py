from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class FrontendSettings(BaseSettings):
    """Runtime settings for the desktop operator UI."""

    backend_url: str = Field(
        default_factory=lambda: os.getenv("BACKEND_URL", "http://localhost:8001")
    )
    request_timeout_sec: float = 1200.0

    model_config = SettingsConfigDict(
        env_prefix="FRONTEND_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@dataclass(frozen=True)
class InferenceOptions:
    mode: str
    sample_fps: float | None
    max_frames: int
    enable_ocr: bool | None
    enable_qr: bool | None
    save_crops: bool


def normalize_backend_url(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    return text.rstrip("/")


def join_url(base: str, path: str) -> str:
    clean_base = normalize_backend_url(base)
    clean_path = path if path.startswith("/") else f"/{path}"
    return f"{clean_base}{clean_path}"


def resolve_download_url(base: str, path: str) -> str:
    text = path.strip()
    if text.startswith("http://") or text.startswith("https://"):
        return text
    if text.startswith("/"):
        return join_url(base, text)
    return join_url(base, f"/{text}")


def bool_to_form(value: bool) -> str:
    return "true" if value else "false"


def build_form_data(options: InferenceOptions) -> dict[str, str]:
    payload: dict[str, str] = {
        "mode": options.mode,
        "max_frames": str(options.max_frames),
        "save_crops": bool_to_form(options.save_crops),
    }
    if options.sample_fps is not None:
        payload["sample_fps"] = str(options.sample_fps)
    if options.enable_ocr is not None:
        payload["enable_ocr"] = bool_to_form(options.enable_ocr)
    if options.enable_qr is not None:
        payload["enable_qr"] = bool_to_form(options.enable_qr)
    return payload


def extract_error_message(response: requests.Response) -> str:
    try:
        body = response.json()
        if isinstance(body, dict) and body.get("detail"):
            return str(body["detail"])
        return json.dumps(body, ensure_ascii=False)
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"


class InferenceWorker(QObject):
    started = Signal(str)
    success = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        backend_url: str,
        video_path: Path,
        options: InferenceOptions,
        timeout_sec: float,
    ) -> None:
        super().__init__()
        self.backend_url = backend_url
        self.video_path = video_path
        self.options = options
        self.timeout_sec = timeout_sec

    def run(self) -> None:
        try:
            if not self.video_path.exists():
                raise FileNotFoundError(f"Video file does not exist: {self.video_path}")

            endpoint = join_url(self.backend_url, "/api/v1/predict/video")
            self.started.emit("Uploading video and running recognition...")

            with self.video_path.open("rb") as stream:
                files = {"file": (self.video_path.name, stream, "video/mp4")}
                data = build_form_data(self.options)
                response = requests.post(
                    endpoint,
                    data=data,
                    files=files,
                    timeout=(15.0, self.timeout_sec),
                )

            if not response.ok:
                raise RuntimeError(extract_error_message(response))

            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("Unexpected backend response format")
            self.success.emit(payload)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self, settings: FrontendSettings) -> None:
        super().__init__()
        self.settings = settings
        self._thread: QThread | None = None
        self._worker: InferenceWorker | None = None
        self._result: dict[str, Any] = {}

        self.setWindowTitle("Lenta Price Vision Operator UI")
        self.resize(1000, 760)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget(self)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        root_layout.addWidget(self._build_connection_box())
        root_layout.addWidget(self._build_video_box())
        root_layout.addWidget(self._build_options_box())
        root_layout.addWidget(self._build_actions_box())
        root_layout.addWidget(self._build_result_box(), stretch=1)
        root_layout.addWidget(self._build_download_box())

        self.setCentralWidget(root)

    def _build_connection_box(self) -> QGroupBox:
        box = QGroupBox("Connection")
        layout = QFormLayout(box)
        self.backend_url_input = QLineEdit(normalize_backend_url(self.settings.backend_url))
        self.backend_url_input.setPlaceholderText("http://localhost:8001")
        layout.addRow("Backend URL", self.backend_url_input)
        return box

    def _build_video_box(self) -> QGroupBox:
        box = QGroupBox("Video")
        layout = QGridLayout(box)

        self.video_path_input = QLineEdit()
        self.video_path_input.setPlaceholderText("Select an .mp4 file")
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._choose_video)

        layout.addWidget(QLabel("Video file"), 0, 0)
        layout.addWidget(self.video_path_input, 0, 1)
        layout.addWidget(browse_button, 0, 2)
        return box

    def _build_options_box(self) -> QGroupBox:
        box = QGroupBox("Inference options")
        layout = QGridLayout(box)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["cpu_safe", "fast", "accurate"])
        self.mode_combo.setCurrentText("cpu_safe")

        self.sample_fps_checkbox = QCheckBox("Override sample FPS")
        self.sample_fps_checkbox.setChecked(True)
        self.sample_fps_spin = QDoubleSpinBox()
        self.sample_fps_spin.setDecimals(2)
        self.sample_fps_spin.setRange(0.1, 60.0)
        self.sample_fps_spin.setValue(2.0)

        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(0, 1_000_000)
        self.max_frames_spin.setValue(0)
        self.max_frames_spin.setToolTip("0 means process all sampled frames")

        self.enable_ocr_checkbox = QCheckBox("Enable OCR")
        self.enable_ocr_checkbox.setChecked(True)
        self.enable_qr_checkbox = QCheckBox("Enable QR")
        self.enable_qr_checkbox.setChecked(True)
        self.save_crops_checkbox = QCheckBox("Save crops")
        self.save_crops_checkbox.setChecked(False)

        layout.addWidget(QLabel("Mode"), 0, 0)
        layout.addWidget(self.mode_combo, 0, 1)
        layout.addWidget(self.sample_fps_checkbox, 0, 2)
        layout.addWidget(self.sample_fps_spin, 0, 3)

        layout.addWidget(QLabel("Max frames"), 1, 0)
        layout.addWidget(self.max_frames_spin, 1, 1)
        layout.addWidget(self.enable_ocr_checkbox, 1, 2)
        layout.addWidget(self.enable_qr_checkbox, 1, 3)
        layout.addWidget(self.save_crops_checkbox, 2, 2)

        return box

    def _build_actions_box(self) -> QGroupBox:
        box = QGroupBox("Run")
        layout = QHBoxLayout(box)
        layout.setSpacing(8)

        self.run_button = QPushButton("Run Recognition")
        self.run_button.clicked.connect(self._start_inference)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)

        self.status_label = QLabel("Idle")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(self.run_button)
        layout.addWidget(self.progress, stretch=1)
        layout.addWidget(self.status_label, stretch=2)
        return box

    def _build_result_box(self) -> QGroupBox:
        box = QGroupBox("Backend response")
        layout = QVBoxLayout(box)

        self.result_view = QTextEdit()
        self.result_view.setReadOnly(True)
        self.result_view.setPlaceholderText("Run inference to see JSON response")

        layout.addWidget(self.result_view)
        return box

    def _build_download_box(self) -> QGroupBox:
        box = QGroupBox("Artifacts")
        layout = QHBoxLayout(box)
        layout.setSpacing(8)

        self.save_csv_button = QPushButton("Save CSV")
        self.save_csv_button.clicked.connect(self._save_csv)
        self.save_csv_button.setEnabled(False)

        self.save_debug_button = QPushButton("Save Debug JSON")
        self.save_debug_button.clicked.connect(self._save_debug)
        self.save_debug_button.setEnabled(False)

        layout.addWidget(self.save_csv_button)
        layout.addWidget(self.save_debug_button)
        layout.addStretch(1)
        return box

    def _choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video",
            "",
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*.*)",
        )
        if path:
            self.video_path_input.setText(path)

    def _start_inference(self) -> None:
        if self._thread is not None:
            QMessageBox.information(self, "In progress", "Inference is already running.")
            return

        backend_url = normalize_backend_url(self.backend_url_input.text())
        if not backend_url:
            QMessageBox.warning(self, "Validation error", "Backend URL is required.")
            return

        video_path = Path(self.video_path_input.text().strip())
        if not video_path.exists():
            QMessageBox.warning(self, "Validation error", "Select an existing video file.")
            return

        options = InferenceOptions(
            mode=self.mode_combo.currentText(),
            sample_fps=self.sample_fps_spin.value()
            if self.sample_fps_checkbox.isChecked()
            else None,
            max_frames=self.max_frames_spin.value(),
            enable_ocr=self.enable_ocr_checkbox.isChecked(),
            enable_qr=self.enable_qr_checkbox.isChecked(),
            save_crops=self.save_crops_checkbox.isChecked(),
        )

        self._set_running_state(True)
        self.status_label.setText("Starting...")

        thread = QThread(self)
        worker = InferenceWorker(
            backend_url=backend_url,
            video_path=video_path,
            options=options,
            timeout_sec=self.settings.request_timeout_sec,
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.started.connect(self.status_label.setText)
        worker.success.connect(self._on_success)
        worker.failed.connect(self._on_failure)

        worker.success.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)

        self._thread = thread
        self._worker = worker
        thread.start()

    def _on_success(self, payload: object) -> None:
        if not isinstance(payload, dict):
            self._on_failure("Unexpected backend response type")
            return
        self._result = payload
        self.result_view.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))
        has_csv_download = bool(self._get_download_url("backend_download", "download"))
        self.save_csv_button.setEnabled(has_csv_download)
        self.save_debug_button.setEnabled(
            bool(self._get_download_url("backend_debug_download", "debug_download"))
        )
        rows = payload.get("rows")
        rows_text = str(rows) if rows is not None else "n/a"
        self.status_label.setText(f"Done. Rows: {rows_text}")

    def _on_failure(self, message: str) -> None:
        self.status_label.setText("Failed")
        QMessageBox.critical(self, "Inference error", message)

    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self._set_running_state(False)

    def _set_running_state(self, running: bool) -> None:
        self.run_button.setEnabled(not running)
        self.progress.setRange(0, 0 if running else 1)
        self.progress.setValue(0 if running else 1)

    def _get_download_url(self, primary_key: str, fallback_key: str) -> str:
        backend_url = normalize_backend_url(self.backend_url_input.text())
        primary = str(self._result.get(primary_key, "")).strip()
        if primary:
            return resolve_download_url(backend_url, primary)
        fallback = str(self._result.get(fallback_key, "")).strip()
        if fallback:
            return resolve_download_url(backend_url, fallback)
        return ""

    def _save_csv(self) -> None:
        self._download_artifact(
            title="Save CSV",
            default_name="recognized.csv",
            url=self._get_download_url("backend_download", "download"),
        )

    def _save_debug(self) -> None:
        self._download_artifact(
            title="Save debug JSON",
            default_name="debug.json",
            url=self._get_download_url("backend_debug_download", "debug_download"),
        )

    def _download_artifact(self, title: str, default_name: str, url: str) -> None:
        if not url:
            QMessageBox.warning(self, "No file", "Download URL is missing in the backend response.")
            return
        target_path, _ = QFileDialog.getSaveFileName(self, title, default_name, "All files (*.*)")
        if not target_path:
            return
        try:
            response = requests.get(url, timeout=(10.0, self.settings.request_timeout_sec))
            if not response.ok:
                raise RuntimeError(extract_error_message(response))
            Path(target_path).write_bytes(response.content)
            QMessageBox.information(self, "Saved", f"File saved to:\n{target_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Download error", str(exc))


def run() -> None:
    app = QApplication([])
    app.setApplicationName("Lenta Price Vision")
    settings = FrontendSettings()
    window = MainWindow(settings)
    window.show()
    app.exec()
