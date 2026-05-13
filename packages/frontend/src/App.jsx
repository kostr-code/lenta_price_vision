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

export default function App() {
  const [backendUrl, setBackendUrl] = useState(DEFAULT_BACKEND_URL);
  const [videoFile, setVideoFile] = useState(null);
  const [mode, setMode] = useState("cpu_safe");
  const [overrideSampleFps, setOverrideSampleFps] = useState(true);
  const [sampleFps, setSampleFps] = useState("2.0");
  const [maxFrames, setMaxFrames] = useState("0");
  const [enableOcr, setEnableOcr] = useState(true);
  const [enableQr, setEnableQr] = useState(true);
  const [saveCrops, setSaveCrops] = useState(false);

  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState("Idle");
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const summary = useMemo(() => {
    if (!result || typeof result !== "object") {
      return { rows: "-", framesSeen: "-", detectionsSeen: "-" };
    }
    return {
      rows: result.rows ?? "-",
      framesSeen: result.frames_seen ?? "-",
      detectionsSeen: result.detections_seen ?? "-",
    };
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
    if (!videoFile) {
      setError("Choose a video file before running recognition.");
      return;
    }

    const parsedMaxFrames = Number.parseInt(maxFrames, 10);
    const safeMaxFrames =
      Number.isFinite(parsedMaxFrames) && parsedMaxFrames >= 0 ? parsedMaxFrames : 0;

    const payload = new FormData();
    payload.append("file", videoFile);
    payload.append("mode", mode);
    payload.append("max_frames", String(safeMaxFrames));
    payload.append("enable_ocr", toFormBoolean(enableOcr));
    payload.append("enable_qr", toFormBoolean(enableQr));
    payload.append("save_crops", toFormBoolean(saveCrops));

    if (overrideSampleFps) {
      const parsedSampleFps = Number.parseFloat(sampleFps);
      if (!Number.isFinite(parsedSampleFps) || parsedSampleFps <= 0) {
        setError("Sample FPS must be a positive number.");
        return;
      }
      payload.append("sample_fps", String(parsedSampleFps));
    }

    setIsRunning(true);
    setStatus("Uploading video and waiting for recognition...");

    try {
      const response = await fetch(joinUrl(normalizedBackend, "/api/v1/predict/video"), {
        method: "POST",
        body: payload,
      });

      if (!response.ok) {
        throw new Error(await parseError(response));
      }

      const body = await response.json();
      setResult(body);
      setStatus(`Done. Rows: ${body.rows ?? "n/a"}`);
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
            Upload robot video, run recognition through backend, and download generated artifacts.
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
              <span>Video file</span>
              <input
                type="file"
                accept="video/mp4,video/*"
                onChange={(event) => setVideoFile(event.target.files?.[0] ?? null)}
                required
              />
            </label>
            <p className="hint">Selected: {videoFile ? videoFile.name : "none"}</p>
          </section>

          <section className="panel">
            <h2>Inference</h2>
            <div className="field two-cols">
              <label>
                <span>Mode</span>
                <select value={mode} onChange={(event) => setMode(event.target.value)}>
                  <option value="cpu_safe">cpu_safe</option>
                  <option value="fast">fast</option>
                  <option value="accurate">accurate</option>
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
                />
              </label>
            </div>

            <label className="toggle-row">
              <input
                type="checkbox"
                checked={overrideSampleFps}
                onChange={(event) => setOverrideSampleFps(event.target.checked)}
              />
              <span>Override sample FPS</span>
            </label>

            <label className="field">
              <span>Sample FPS</span>
              <input
                type="number"
                min="0.1"
                step="0.1"
                value={sampleFps}
                onChange={(event) => setSampleFps(event.target.value)}
                disabled={!overrideSampleFps}
              />
            </label>

            <div className="toggle-grid">
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={enableOcr}
                  onChange={(event) => setEnableOcr(event.target.checked)}
                />
                <span>Enable OCR</span>
              </label>
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={enableQr}
                  onChange={(event) => setEnableQr(event.target.checked)}
                />
                <span>Enable QR</span>
              </label>
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={saveCrops}
                  onChange={(event) => setSaveCrops(event.target.checked)}
                />
                <span>Save crops</span>
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
                <p className="metric-label">Frames Seen</p>
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

        <section className="panel full-width">
          <h2>Backend Response</h2>
          <pre className="json-view">{hasResult ? prettyJson(result) : "No result yet."}</pre>
        </section>
      </main>
    </div>
  );
}
