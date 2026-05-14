from backend.main import enrich_with_backend_downloads, parse_download_parts, parse_origins


def test_parse_origins_star() -> None:
    assert parse_origins("*") == ["*"]


def test_parse_origins_csv() -> None:
    assert parse_origins("http://localhost:3000, https://demo.example.com ") == [
        "http://localhost:3000",
        "https://demo.example.com",
    ]


def test_parse_download_parts() -> None:
    parsed = parse_download_parts("/download/run_abc123/results.csv")
    assert parsed is not None
    assert parsed.run_id == "run_abc123"
    assert parsed.filename == "results.csv"


def test_enrich_with_backend_downloads() -> None:
    payload = {
        "download": "/download/run_abc123/recognized.csv",
        "debug_download": "/download/run_abc123/debug.json",
    }
    enrich_with_backend_downloads(payload)

    assert payload["backend_download"] == "/api/v1/download/run_abc123/recognized.csv"
    assert payload["backend_debug_download"] == "/api/v1/download/run_abc123/debug.json"
