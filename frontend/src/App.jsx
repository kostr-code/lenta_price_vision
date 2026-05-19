import { useMemo, useState } from "react";

const DEFAULT_BACKEND_URL = (import.meta.env.VITE_BACKEND_URL ?? "http://localhost:8001").trim();

function normalizeBackendUrl(value) {
  return value.trim().replace(/\/+$/, "");
}

function joinUrl(base, path) {
  const normalized = normalizeBackendUrl(base);
  const suffix = path.startsWith("/") ? path : `/${path}`;
  return `${normalized}${suffix}`;
}

function toFormBoolean(value) {
  return value ? "true" : "false";
}

async function parseError(response) {
  try {
    const body = await response.json();
    if (body && typeof body === "object" && body.detail) {
      return String(body.detail);
    }
    return JSON.stringify(body);
  } catch {
    const text = await response.text();
    return text || `HTTP ${response.status}`;
  }
}

function resolveDownloadUrl(base, value) {
  const text = String(value ?? "").trim();
  if (!text) {
    return "";
  }
  if (text.startsWith("http://") || text.startsWith("https://")) {
    return text;
  }
  return joinUrl(base, text.startsWith("/") ? text : `/${text}`);
}

function prettyJson(value) {
  return JSON.stringify(value, null, 2);
}

function isImageFile(file) {
  if (!file) {
    return false;
  }
  const contentType = String(file.type ?? "").toLowerCase();
  if (contentType.startsWith("image/")) {
    return true;
  }
  const name = String(file.name ?? "").toLowerCase();
  return [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"].some((suffix) =>
    name.endsWith(suffix),
  );
}

export default function App() {
  const [backendUrl, setBackendUrl] = useState(DEFAULT_BACKEND_URL);
  const [mediaFile, setMediaFile] = useState(null);
  const [mode, setMode] = useState("accurate");
  const [sampleFps, setSampleFps] = useState("2.0");
  const [maxFrames, setMaxFrames] = useState("0");
  const [enableQr, setEnableQr] = useState(true);

  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState("Idle");
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const selectedIsImage = isImageFile(mediaFile);

  const summary = useMemo(() => {
    if (!result || typeof result !== "object") {
      return { rows: "-", framesSeen: "-", detectionsSeen: "-" };
    }
    return {
      rows: result.row_count ?? "-",
      framesSeen: result.frames_seen ?? "-",
      detectionsSeen: result.detections_seen ?? "-",
    };
  }, [result]);

  const cropRows = useMemo(() => {
    if (!result || !Array.isArray(result.rows)) return [];
    return result.rows.filter((r) => r && typeof r === "object");
  }, [result]);

  const hasResult = Boolean(result && typeof result === "object");

  async function handleRun(event) {
    event.preventDefault();
    setError("");

    const normalizedBackend = normalizeBackendUrl(backendUrl);
    if (!normalizedBackend) {
      setError("Backend URL is required.");
      return;
    }
    if (!mediaFile) {
      setError("Choose an image or video file before running recognition.");
      return;
    }

    const isImageUpload = isImageFile(mediaFile);

    const payload = new FormData();
    payload.append("file", mediaFile);
    payload.append("mode", mode);
    payload.append("enable_qr", toFormBoolean(enableQr));

    if (!isImageUpload) {
      const parsedMaxFrames = Number.parseInt(maxFrames, 10);
      const safeMaxFrames =
        Number.isFinite(parsedMaxFrames) && parsedMaxFrames >= 0 ? parsedMaxFrames : 0;
      payload.append("max_frames", String(safeMaxFrames));

      const parsedSampleFps = Number.parseFloat(sampleFps);
      if (Number.isFinite(parsedSampleFps) && parsedSampleFps > 0) {
        payload.append("sample_fps", String(parsedSampleFps));
      }
    }

    setIsRunning(true);
    setStatus(
      isImageUpload
        ? "Uploading image and waiting for recognition..."
        : "Uploading video and waiting for recognition...",
    );

    try {
      const endpoint = isImageUpload ? "/api/v1/predict/image" : "/api/v1/predict/video";
      const response = await fetch(joinUrl(normalizedBackend, endpoint), {
        method: "POST",
        body: payload,
      });

      if (!response.ok) {
        throw new Error(await parseError(response));
      }

      const body = await response.json();
      setResult(body);
      setStatus(`Done. Rows: ${body.row_count ?? "n/a"}`);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Unexpected error";
      setStatus("Failed");
      setError(message);
    } finally {
      setIsRunning(false);
    }
  }

  async function handleDownload(primaryKey, fallbackKey, defaultFileName) {
    if (!result || typeof result !== "object") {
      return;
    }

    const url = resolveDownloadUrl(backendUrl, result[primaryKey] || result[fallbackKey] || "");
    if (!url) {
      setError("No download link in backend response.");
      return;
    }

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(await parseError(response));
      }
      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = objectUrl;
      link.download = defaultFileName;
      link.click();
      URL.revokeObjectURL(objectUrl);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Download failed";
      setError(message);
    }
  }

  return (
    <div className="page-shell">
      <div className="background-shape background-shape-a" />
      <div className="background-shape background-shape-b" />
      <main className="app">
        <header className="hero">
          <p className="eyebrow">Lenta Price Vision</p>
          <h1>React Operator Console</h1>
          <p className="hero-copy">
            Upload a shelf photo or robot video, run recognition through backend, and download
            generated artifacts.
          </p>
        </header>

        <form className="panel-grid" onSubmit={handleRun}>
          <section className="panel">
            <h2>Connection</h2>
            <label className="field">
              <span>Backend URL</span>
              <input
                type="url"
                value={backendUrl}
                onChange={(event) => setBackendUrl(event.target.value)}
                placeholder="http://localhost:8001"
                required
              />
            </label>

            <label className="field">
              <span>Image or video file</span>
              <input
                type="file"
                accept="image/*,video/mp4,video/*,.jpg,.jpeg,.png,.bmp,.webp,.tif,.tiff"
                onChange={(event) => setMediaFile(event.target.files?.[0] ?? null)}
                required
              />
            </label>
            <p className="hint">
              Selected: {mediaFile ? mediaFile.name : "none"}{" "}
              {selectedIsImage ? "(image mode)" : mediaFile ? "(video mode)" : ""}
            </p>
          </section>

          <section className="panel">
            <h2>Inference</h2>
            <div className="field two-cols">
              <label>
                <span>Mode</span>
                <select value={mode} onChange={(event) => setMode(event.target.value)}>
                  <option value="accurate">VLM + OCR (полный)</option>
                  <option value="cpu_safe">Только OCR (быстрый)</option>
                </select>
              </label>
              <label>
                <span>Max frames</span>
                <input
                  type="number"
                  min="0"
                  step="1"
                  value={maxFrames}
                  onChange={(event) => setMaxFrames(event.target.value)}
                  disabled={selectedIsImage}
                />
              </label>
            </div>

            <label className="field">
              <span>Sample FPS</span>
              <input
                type="number"
                min="0"
                step="0.1"
                value={sampleFps}
                onChange={(event) => setSampleFps(event.target.value)}
                disabled={selectedIsImage}
              />
            </label>
            <p className="hint">
              {selectedIsImage
                ? "Photo selected: frame sampling options are skipped."
                : "0 = без ограничений (все кадры)"}
            </p>

            <div className="toggle-grid">
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={enableQr}
                  onChange={(event) => setEnableQr(event.target.checked)}
                />
                <span>Enable QR</span>
              </label>
            </div>

            <button className="run-button" type="submit" disabled={isRunning}>
              {isRunning ? "Running..." : "Run Recognition"}
            </button>
            <p className="status">Status: {status}</p>
            {error ? <p className="error">{error}</p> : null}
          </section>

          <section className="panel">
            <h2>Summary</h2>
            <div className="metric-grid">
              <div>
                <p className="metric-label">Rows</p>
                <p className="metric-value">{summary.rows}</p>
              </div>
              <div>
                <p className="metric-label">Frames/Images Seen</p>
                <p className="metric-value">{summary.framesSeen}</p>
              </div>
              <div>
                <p className="metric-label">Detections Seen</p>
                <p className="metric-value">{summary.detectionsSeen}</p>
              </div>
            </div>

            <div className="download-row">
              <button
                type="button"
                className="ghost-button"
                disabled={!hasResult}
                onClick={() => handleDownload("backend_download", "download", "recognized.csv")}
              >
                Download CSV
              </button>
              <button
                type="button"
                className="ghost-button"
                disabled={!hasResult}
                onClick={() =>
                  handleDownload("backend_debug_download", "debug_download", "debug.json")
                }
              >
                Download Debug JSON
              </button>
            </div>
          </section>
        </form>

        {cropRows.length > 0 && (
          <section className="panel full-width" style={{ marginTop: "14px" }}>
            <h2>Распознанные ценники</h2>
            <div className="crop-grid">
              {cropRows.map((row, i) => {
                const imgSrc = row._crop_url
                  ? resolveDownloadUrl(backendUrl, row._crop_url)
                  : "";
                return (
                  <div key={i} className="crop-card">
                    {imgSrc ? (
                      <img src={imgSrc} alt={`ценник ${i + 1}`} loading="lazy" />
                    ) : (
                      <div className="crop-placeholder" />
                    )}
                    <div className="crop-card-body">
                      <p className="crop-name">{row.product_name || "—"}</p>
                      <p className="crop-prices">
                        {row.price_card || "—"}
                        {row.price_default ? ` / ${row.price_default}` : ""}
                      </p>
                      <p className="crop-meta">
                        {row.color ? (
                          <span className="crop-color">{row.color}</span>
                        ) : null}
                        {row._quality ? (
                          <span className="crop-quality">q: {row._quality}</span>
                        ) : null}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        )}

        <section className="panel full-width" style={{ marginTop: "14px" }}>
          <h2>Backend Response</h2>
          <pre className="json-view">{hasResult ? prettyJson(result) : "No result yet."}</pre>
        </section>
      </main>
    </div>
  );
}
