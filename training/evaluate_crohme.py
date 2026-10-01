"""Phase 15: evaluate the frozen MathVision pipeline on real CROHME test data.

Scope: CROHME2019_test expressions whose SymLG labels are single-line V1
(digits,+,-,=,x,/,times with only Right relations). Metrics:
  symbol accuracy (order-free multiset match vs lg labels),
  equation accuracy (predicted string == normalized LaTeX truth),
  solution accuracy (SymPy verdict agreement).
Writes data/processed/crohme_eval.json. Read-only wrt raw data.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "raw" / "crohme" / "TC11_CROHME23"
OUT = ROOT / "data" / "processed" / "crohme_eval.json"

import sys
sys.path.insert(0, str(ROOT))
from app.main import run_pipeline  # noqa: E402
from app.solver.equation_solver import solve_expression  # noqa: E402

V1 = set("0123456789+-=x") | {"times", "div", "/"}
LG_MAP = {"times": "\u00d7", "div": "\u00f7", "/": "\u00f7"}
TRUTH_RE = re.compile(r'<annotation type="truth">\s*(.*?)\s*</annotation>', re.S)


def lg_info(lg: Path):
    txt = lg.read_text(encoding="utf-8", errors="replace")
    labels = [ln.split(",")[2].strip() for ln in txt.splitlines() if ln.startswith("O,")]
    rels = {ln.split(",")[3].strip() for ln in txt.splitlines() if ln.startswith("R,")}
    return labels, rels


def norm_truth(t: str) -> str:
    t = t.replace("$", "").replace(" ", "").replace("\\times", "\u00d7")
    t = t.replace("\\div", "\u00f7").replace("\\cdot", "\u00d7")
    return t


def main(max_exprs: int = 50) -> None:
    lgdir = TEST / "SymLG" / "test" / "CROHME2019_test"
    imgdir = TEST / "IMG" / "test" / "CROHME2019_test"
    inkdir = TEST / "INKML" / "test" / "CROHME2019_test"
    cands = []
    for lg in sorted(lgdir.glob("*.lg")):
        labels, rels = lg_info(lg)
        if (labels and set(labels) <= V1 and rels <= {"Right"}
                and (imgdir / f"{lg.stem}.png").exists()
                and (inkdir / f"{lg.stem}.inkml").exists()):
            cands.append((lg.stem, labels))
        if len(cands) >= max_exprs:
            break
    print(f"[crohme-eval] V1 single-line candidates: {len(cands)}")

    sym_n = sym_d = eq_ok = sol_ok = n = 0
    per_expr = []
    for stem, labels in cands:
        img = cv2.imread(str(imgdir / f"{stem}.png"), cv2.IMREAD_COLOR)
        if img is None:
            continue
        truth = TRUTH_RE.search((inkdir / f"{stem}.inkml").read_text(
            encoding="utf-8", errors="replace")).group(1)
        exp_truth = norm_truth(truth)
        _b, _c, details, pred, res = run_pipeline(img)
        want = Counter(LG_MAP.get(lbl, lbl) for lbl in labels)
        got = Counter(lab for lab, _ in details)
        hit = sum((want & got).values())
        sym_n += hit
        sym_d += sum(want.values())
        eq_match = (pred == exp_truth)
        eq_ok += eq_match
        # solution agreement: solve both strings
        r_pred = solve_expression(pred) if pred else None
        r_true = solve_expression(exp_truth)
        agree = False
        if r_pred is not None and r_pred.kind == r_true.kind:
            agree = (r_pred.message == r_true.message
                     or r_pred.data.get("correct") == r_true.data.get("correct"))
        sol_ok += agree
        n += 1
        per_expr.append({"id": stem, "truth": exp_truth, "pred": pred,
                         "eq_match": eq_match, "sol_agree": agree,
                         "conf": round(float(sum(c for _, c in details) / max(1, len(details))), 3)})
    report = {"n": n,
              "symbol_accuracy": sym_n / max(1, sym_d),
              "equation_accuracy": eq_ok / max(1, n),
              "solution_accuracy": sol_ok / max(1, n),
              "expressions": per_expr}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[crohme-eval] n={n} symbol={report['symbol_accuracy']:.3f} "
          f"equation={report['equation_accuracy']:.3f} solution={report['solution_accuracy']:.3f}")
    print(f"[crohme-eval] -> {OUT}")


if __name__ == "__main__":
    main()
