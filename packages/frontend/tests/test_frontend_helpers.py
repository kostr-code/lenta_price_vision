from frontend.main import (
    InferenceOptions,
    build_form_data,
    join_url,
    normalize_backend_url,
    resolve_download_url,
)


def test_normalize_backend_url() -> None:
    assert normalize_backend_url(" http://localhost:8001/ ") == "http://localhost:8001"


def test_join_url() -> None:
    assert join_url("http://localhost:8001/", "/api/v1/schema") == "http://localhost:8001/api/v1/schema"
    assert join_url("http://localhost:8001", "api/v1/schema") == "http://localhost:8001/api/v1/schema"


def test_resolve_download_url_relative() -> None:
    assert (
        resolve_download_url("http://localhost:8001", "/api/v1/download/run1/file.csv")
        == "http://localhost:8001/api/v1/download/run1/file.csv"
    )


def test_resolve_download_url_absolute() -> None:
    assert (
        resolve_download_url("http://localhost:8001", "https://example.com/file.csv")
        == "https://example.com/file.csv"
    )


def test_build_form_data() -> None:
    data = build_form_data(
        InferenceOptions(
            mode="cpu_safe",
            sample_fps=2.0,
            max_frames=0,
            enable_ocr=True,
            enable_qr=False,
            save_crops=False,
        )
    )
    assert data["mode"] == "cpu_safe"
    assert data["sample_fps"] == "2.0"
    assert data["max_frames"] == "0"
    assert data["enable_ocr"] == "true"
    assert data["enable_qr"] == "false"
    assert data["save_crops"] == "false"
