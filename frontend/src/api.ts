export interface SymbolConf { label: string; confidence: number }

export interface PredictResult {
  expression: string;
  solution_kind: string;
  solution_message: string;
  confidence: number;
  ms: number;
  num_symbols: number;
  symbols: SymbolConf[];
}

const BASE = "http://localhost:8000";

export async function getHealth(): Promise<{ online: boolean }> {
  try {
    const r = await fetch(`${BASE}/api/health`);
    if (!r.ok) return { online: false };
    const j = await r.json();
    return { online: !!j.online };
  } catch {
    return { online: false };
  }
}

export async function predictImage(blob: Blob): Promise<PredictResult> {
  const fd = new FormData();
  fd.append("file", blob, "equation.png");
  const r = await fetch(`${BASE}/api/predict`, { method: "POST", body: fd });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || "Inference failed.");
  if (j.solution_kind === "error" && j.num_symbols === 0)
    throw new Error(j.solution_message || "No symbols detected.");
  return j as PredictResult;
}

/** Initial visual state only (mirrors the reference mockup). Replaced by real API output on Solve. */
export const EXAMPLE_RESULT: PredictResult = {
  expression: "2x + 5 = 15",
  solution_kind: "linear_solution",
  solution_message: "x = 5",
  confidence: 0.99,
  ms: 669,
  num_symbols: 7,
  symbols: [
    { label: "2", confidence: 0.978 },
    { label: "x", confidence: 0.942 },
    { label: "+", confidence: 0.991 },
    { label: "5", confidence: 0.986 },
    { label: "=", confidence: 0.994 },
    { label: "1", confidence: 0.989 },
    { label: "5", confidence: 0.975 },
  ],
};
