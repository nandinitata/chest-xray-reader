import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

function confidenceBand(p) {
  if (p >= 0.75) return "likely";
  if (p >= 0.6) return "possible";
  return "borderline";
}

function TopBar({ device }) {
  return (
    <header className="topbar">
      <div className="topbar__mark">
        <span className="topbar__glyph">▚</span>
        <span className="topbar__title">READING ROOM</span>
        <span className="topbar__sub">chest film reader</span>
      </div>
      <div className="topbar__status">
        <span className={`dot ${device ? "dot--live" : "dot--idle"}`} />
        <span className="topbar__device">{device ? `compute · ${device}` : "offline"}</span>
      </div>
    </header>
  );
}

function Lightbox({ previewUrl, heatmapUrl, view, analyzing, onPick }) {
  const inputRef = useRef(null);
  const showHeat = view === "heatmap" && heatmapUrl;

  const onDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) onPick(file);
  };

  return (
    <section
      className="lightbox"
      onDragOver={(e) => e.preventDefault()}
      onDrop={onDrop}
    >
      <div className="lightbox__frame">
        {previewUrl ? (
          <div className="film">
            <img className="film__img" src={previewUrl} alt="chest film" />
            <img
              className={`film__img film__img--heat ${showHeat ? "is-on" : ""}`}
              src={heatmapUrl || previewUrl}
              alt="heat map"
              aria-hidden={!showHeat}
            />
            {analyzing && <div className="scanline" />}
          </div>
        ) : (
          <button className="dropzone" onClick={() => inputRef.current?.click()}>
            <span className="dropzone__cue">Drop a chest X-ray</span>
            <span className="dropzone__hint">or click to load a film · PNG / JPG</span>
          </button>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => e.target.files?.[0] && onPick(e.target.files[0])}
      />
    </section>
  );
}

function FindingRow({ f, index }) {
  return (
    <li className="finding" style={{ animationDelay: `${120 + index * 90}ms` }}>
      <div className="finding__head">
        <span className="finding__name">{f.plain}</span>
        <span className={`chip chip--${confidenceBand(f.probability)}`}>
          {confidenceBand(f.probability)}
        </span>
      </div>
      <div className="finding__meter">
        <span
          className="finding__fill"
          style={{ width: `${Math.round(f.probability * 100)}%` }}
        />
        <span className="finding__pct">{(f.probability * 100).toFixed(0)}</span>
      </div>
      <span className="finding__label">{f.label}</span>
    </li>
  );
}

function ReportCard({ report }) {
  if (!report) return null;
  const sourceLabel =
    report.source === "ollama"
      ? "written by the local model"
      : report.source === "template"
      ? "auto-summary"
      : "note";
  return (
    <div className={`report ${report.refused ? "report--refused" : ""}`}>
      <div className="report__bar">
        <span className="report__tag">REPORT</span>
        <span className="report__source">{sourceLabel}</span>
      </div>
      <p className="report__body">{report.text}</p>
      <p className="report__disclaimer">{report.disclaimer}</p>
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [view, setView] = useState("heatmap");
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState(null);
  const [device, setDevice] = useState(null);

  const onPick = useCallback((f) => {
    setFile(f);
    setResult(null);
    setError(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(f));
  }, [previewUrl]);

  const analyze = useCallback(async () => {
    if (!file) return;
    setAnalyzing(true);
    setError(null);
    try {
      const body = new FormData();
      body.append("image", file);
      const res = await fetch(`${API_BASE}/analyze`, { method: "POST", body });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Server said ${res.status}`);
      }
      const data = await res.json();
      setResult(data);
      setView(data.heatmap_data_uri ? "heatmap" : "original");
    } catch (e) {
      setError(e.message);
    } finally {
      setAnalyzing(false);
    }
  }, [file]);

  // Show whether the backend is up and what it's computing on.
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((r) => r.json())
      .then((d) => setDevice(d.device))
      .catch(() => setDevice(null));
  }, []);

  // "See a demo" mode: load a pre-computed reading so a visitor can look at the
  // output without downloading a chest X-ray first. Add ?demo to the URL.
  useEffect(() => {
    if (!new URLSearchParams(window.location.search).has("demo")) return;
    (async () => {
      try {
        const data = await (await fetch("/demo.json")).json();
        setPreviewUrl("/demo.png");
        setResult(data);
        setView(data.heatmap_data_uri ? "heatmap" : "original");
      } catch {
        /* no demo asset, ignore */
      }
    })();
  }, []);

  const refused = result && result.is_chest_xray === false;
  const heatmapUrl = result?.heatmap_data_uri || null;

  const summaryLine = useMemo(() => {
    if (!result) return "No film loaded.";
    if (refused) return "Not a chest film.";
    const n = result.findings.length;
    if (n === 0) return "Reads as unremarkable.";
    return `${n} finding${n > 1 ? "s" : ""} above threshold.`;
  }, [result, refused]);

  return (
    <div className="app">
      <TopBar device={device} />

      <main className="stage">
        <div className="stage__left">
          <Lightbox
            previewUrl={previewUrl}
            heatmapUrl={heatmapUrl}
            view={view}
            analyzing={analyzing}
            onPick={onPick}
          />
          <div className="controls">
            <div className="toggle" role="group" aria-label="view">
              <button
                className={`toggle__btn ${view === "original" ? "is-active" : ""}`}
                onClick={() => setView("original")}
                disabled={!previewUrl}
              >
                Film
              </button>
              <button
                className={`toggle__btn ${view === "heatmap" ? "is-active" : ""}`}
                onClick={() => setView("heatmap")}
                disabled={!heatmapUrl}
              >
                Heat map
              </button>
            </div>
            <button
              className="read-btn"
              onClick={analyze}
              disabled={!file || analyzing}
            >
              {analyzing ? "Reading…" : "Read film"}
            </button>
          </div>
        </div>

        <aside className="stage__right">
          <div className="panel-head">
            <span className="panel-head__eyebrow">READ-OUT</span>
            <span className="panel-head__summary">{summaryLine}</span>
          </div>

          {error && <div className="alert">{error}</div>}

          {refused && (
            <div className="refusal">
              <span className="refusal__mark">⦸</span>
              <p>{result.ood_reason}</p>
            </div>
          )}

          {result && !refused && (
            <>
              <ul className="findings">
                {result.findings.length === 0 && (
                  <li className="finding finding--empty">
                    Nothing crossed the reporting line. The film reads clean.
                  </li>
                )}
                {result.findings.map((f, i) => (
                  <FindingRow key={f.label} f={f} index={i} />
                ))}
              </ul>
              <ReportCard report={result.report} />
            </>
          )}

          {!result && !error && (
            <div className="empty-note">
              Load a film and hit <b>Read film</b>. Findings and a short report
              show up here, with a heat map over the image.
            </div>
          )}
        </aside>
      </main>
    </div>
  );
}
