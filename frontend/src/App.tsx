import { useEffect, useRef, useState } from "react";
import { EXAMPLE_RESULT, getHealth, predictImage, type PredictResult } from "./api";

type Mode = "draw" | "upload";

const INK = "#17191b";
const PAPER = "#f4f0e6";

export default function App() {
  const [mode, setMode] = useState<Mode>("draw");
  const [result, setResult] = useState<PredictResult>(EXAMPLE_RESULT);
  const [isExample, setIsExample] = useState(true);
  const [online, setOnline] = useState(false);
  const [solving, setSolving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadURL, setUploadURL] = useState<string | null>(null);
  const [hasInk, setHasInk] = useState(true); // example handwriting visible initially
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const last = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    getHealth().then((h) => setOnline(h.online));
    seedExample();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function canvas2d() {
    const c = canvasRef.current;
    return c ? c.getContext("2d") : null;
  }

  function seedExample() {
    const c = canvasRef.current;
    const ctx = c?.getContext("2d");
    if (!c || !ctx) return;
    ctx.fillStyle = PAPER;
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.fillStyle = INK;
    ctx.font = `500 ${Math.floor(c.height * 0.52)}px "Segoe Script", "Brush Script MT", cursive`;
    ctx.textBaseline = "middle";
    ctx.fillText("2x+5=15", c.width * 0.06, c.height * 0.52);
    ctx.fillStyle = "rgba(0,0,0,0.35)";
    ctx.font = `400 ${Math.floor(c.height * 0.11)}px serif`;
    ctx.fillText("∫  √  π", c.width * 0.045, c.height * 0.86);
  }

  function pos(e: React.PointerEvent) {
    const c = canvasRef.current!;
    const r = c.getBoundingClientRect();
    return {
      x: ((e.clientX - r.left) / r.width) * c.width,
      y: ((e.clientY - r.top) / r.height) * c.height,
    };
  }

  function clearPaper() {
    const c = canvasRef.current;
    const ctx = canvas2d();
    if (!c || !ctx) return;
    ctx.fillStyle = PAPER;
    ctx.fillRect(0, 0, c.width, c.height);
  }

  function onDown(e: React.PointerEvent) {
    if (mode !== "draw") return;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    if (hasInk && isExample) {
      // first real stroke clears the example handwriting
      clearPaper();
      setIsExample(false);
    }
    drawing.current = true;
    last.current = pos(e);
    setHasInk(true);
  }

  function onMove(e: React.PointerEvent) {
    if (!drawing.current || mode !== "draw") return;
    const ctx = canvas2d();
    if (!ctx) return;
    const p = pos(e);
    const l = last.current ?? p;
    ctx.strokeStyle = INK;
    ctx.lineWidth = 7;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(l.x, l.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    last.current = p;
    setIsExample(false);
  }

  function onUp() {
    drawing.current = false;
    last.current = null;
  }

  function onClear() {
    clearPaper();
    setHasInk(false);
    setIsExample(false);
    setError(null);
  }

  async function onSolve() {
    setError(null);
    try {
      let blob: Blob | null = null;
      if (mode === "draw") {
        const c = canvasRef.current;
        if (!c || !hasInk) {
          setError("Draw or upload an equation first.");
          return;
        }
        blob = await new Promise<Blob | null>((res) => c.toBlob(res, "image/png"));
      } else {
        blob = uploadFile;
        if (!blob) {
          setError("Drop an equation image above to begin.");
          return;
        }
      }
      if (!blob) {
        setError("Draw or upload an equation first.");
        return;
      }
      setSolving(true);
      const out = await predictImage(blob);
      setResult(out);
      setIsExample(false);
      if (out.solution_kind === "error") setError(out.solution_message);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not process that image.");
    } finally {
      setSolving(false);
    }
  }

  function onFile(f: File | undefined) {
    setError(null);
    if (!f) return;
    if (!/\.(png|jpe?g)$/i.test(f.name)) {
      setError("Unsupported file — use PNG, JPG or JPEG.");
      return;
    }
    setUploadFile(f);
    setUploadURL((u) => {
      if (u) URL.revokeObjectURL(u);
      return URL.createObjectURL(f);
    });
  }

  const ready = mode === "draw" ? hasInk : !!uploadFile;

  return (
    <div className="mv-root">
      {/* ---------- header ---------- */}
      <header className="mv-header">
        <div className="mv-head-left">
          <span className="mv-logo">Σ MATHVISION</span>
          <span className="mv-tag">HANDWRITTEN MATHEMATICS, UNDERSTOOD.</span>
        </div>
        <div className="mv-head-right">
          <span className="mv-link">About</span>
          <span className="mv-link">Model</span>
          <span className={`mv-online ${online ? "on" : "off"}`}>
            <i /> {online ? "ONLINE" : "OFFLINE"}
          </span>
        </div>
      </header>

      {/* ---------- top workspace ---------- */}
      <main className="mv-grid-top">
        {/* left: input */}
        <section className="mv-panel">
          <div className="mv-panel-head">
            <span>WRITE YOUR EQUATION</span>
            <div className="mv-switch" role="tablist" aria-label="Input mode">
              <button
                role="tab"
                aria-selected={mode === "draw"}
                className={mode === "draw" ? "active" : ""}
                onClick={() => setMode("draw")}
              >
                Draw
              </button>
              <button
                role="tab"
                aria-selected={mode === "upload"}
                className={mode === "upload" ? "active" : ""}
                onClick={() => setMode("upload")}
              >
                Upload
              </button>
            </div>
          </div>

          {mode === "draw" ? (
            <div className="mv-canvas-wrap">
              <canvas
                ref={canvasRef}
                width={900}
                height={260}
                className="mv-canvas"
                onPointerDown={onDown}
                onPointerMove={onMove}
                onPointerUp={onUp}
                onPointerLeave={onUp}
              />
              {!online && <div className="mv-canvas-note">Backend offline — start the API on :8000.</div>}
            </div>
          ) : (
            <label
              className="mv-upload"
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                onFile(e.dataTransfer.files?.[0]);
              }}
            >
              {uploadURL ? (
                <img src={uploadURL} alt="Upload preview" className="mv-upload-img" />
              ) : (
                <div className="mv-upload-empty">
                  <div className="mv-upload-title">Drop an equation image here</div>
                  <div className="mv-upload-sub">Supported: PNG · JPG · JPEG</div>
                </div>
              )}
              <input
                type="file"
                accept=".png,.jpg,.jpeg"
                hidden
                onChange={(e) => onFile(e.target.files?.[0])}
              />
            </label>
          )}

          <div className="mv-actions">
            <button className="mv-btn mv-btn-ghost" onClick={onClear}>
              Clear
            </button>
            <button
              className="mv-btn mv-btn-primary"
              onClick={onSolve}
              disabled={!ready || solving}
            >
              {solving ? "Analyzing…" : "Solve Equation →"}
            </button>
          </div>
          {error && <div className="mv-error">{error}</div>}
          {solving && (
            <div className="mv-loading">
              Analyzing handwriting… Preprocessing → Detecting → Recognizing → Solving
            </div>
          )}
        </section>

        {/* right: result */}
        <section className="mv-panel">
          <div className="mv-panel-head">
            <span>MATHVISION RESULT</span>
          </div>
          <div className="mv-label">RECOGNIZED EQUATION</div>
          <div className="mv-expr">{result.expression || "—"}</div>
          <div className="mv-result-row">
            <div className="mv-solution">
              <div className="mv-label green">SOLUTION</div>
              <div className="mv-answer">{result.solution_message}</div>
              <div className="mv-sub">solved with SymPy</div>
            </div>
            <div className="mv-stats">
              <div className="mv-stat">
                <div className="mv-label">CONFIDENCE</div>
                <div className="mv-stat-val">{(result.confidence * 100).toFixed(1)}%</div>
              </div>
              <div className="mv-stat">
                <div className="mv-label">INFERENCE</div>
                <div className="mv-stat-val cyan">{result.ms.toFixed(0)} ms</div>
              </div>
            </div>
          </div>
          {isExample && (
            <div className="mv-example-note">Example state — draw and press Solve for live inference.</div>
          )}
        </section>
      </main>

      {/* ---------- recognition + model ---------- */}
      <main className="mv-grid-mid">
        <section className="mv-panel">
          <div className="mv-panel-head">
            <span>RECOGNITION · CNN OUTPUT</span>
          </div>
          <div className="mv-symbols">
            {result.symbols.map((s, i) => (
              <div className="mv-sym" key={`${s.label}-${i}`}>
                <div className="mv-sym-card">{s.label}</div>
                <div className={`mv-sym-bar ${s.confidence < 0.95 ? "cyan" : ""}`}>
                  <span style={{ width: `${(s.confidence * 100).toFixed(1)}%` }} />
                </div>
                <div className="mv-sym-conf">{(s.confidence * 100).toFixed(1)}%</div>
              </div>
            ))}
          </div>
        </section>

        <aside className="mv-panel mv-model">
          <div className="mv-model-title">MATHVISION CNN</div>
          <div className="mv-model-sub">Handwritten symbol classification</div>
          <div className="mv-model-row">
            <span>Input</span>
            <b>28 × 28 × 1</b>
          </div>
          <div className="mv-model-row">
            <span>Classes</span>
            <b>0–9 + − × ÷ = x</b>
          </div>
          <div className="mv-model-row">
            <span>Inference</span>
            <b className="cyan">LOCAL</b>
          </div>
        </aside>
      </main>

      {/* ---------- pipeline ---------- */}
      <footer className="mv-panel mv-pipe-panel">
        <div className="mv-panel-head">
          <span>HOW MATHVISION UNDERSTANDS</span>
        </div>
        <div className="mv-pipe">
          {["INPUT", "PREPROCESS", "SEGMENT", "CNN", "RECONSTRUCT", "SOLVE"].map((s) => (
            <div className="mv-pipe-step" key={s}>
              <div className={`mv-pipe-card ${s === "CNN" ? "hot" : ""}`}>{s}</div>
            </div>
          ))}
        </div>
      </footer>
    </div>
  );
}
