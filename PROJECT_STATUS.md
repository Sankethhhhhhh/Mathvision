# MathVision — Project Status (final, real-data system)

## Phase tracker (master spec numbering)
- Phase 0 — Audit — COMPLETE (Python 3.12.10 venv; TF 2.21/OpenCV/SymPy/Streamlit 1.44/pytest verified; canvas incompatibility root-caused and fixed by pinning streamlit==1.44.0 + drawable-canvas==0.9.3)
- Phase 1 — HASYv2 — COMPLETE (168,233 imgs inspected; exact IDs in `class_mapping.json`; 3,138 filtered samples; `dataset_report.json`; `prepare_hasyv2.py`; `=` proven absent → synthetic/CROHME)
- Phase 2 — CROHME — COMPLETE (27,279 PNG / 164,287 InkML / 172,332 SymLG extracted byte-perfect; `crohme_info.json`; `inspect_crohme.py`; viz in `assets/dataset_samples/crohme/`)
- Phase 3 — CNN (hybrid HASY) — COMPLETE (`build_hasy_dataset.py`: MNIST digits + HASY +−×÷ + CROHME x/= + synth; file/grouped splits; 22,400/4,800/4,800; `models/mathvision_cnn.keras`)
- Phase 4 — Evaluation — COMPLETE (test **96.48%**, macro F1 0.9650, x-recall 98.7%; `metrics.json` shown live in UI)
- Phase 5 — Preprocessing — COMPLETE (+ adaptive closing, stroke-width normalization for hairline scans)
- Phase 6 — Segmentation — COMPLETE (+ fragment absorption; scale tests 2×/3×/5× pass)
- Phase 7 — Integration — COMPLETE (cached predictor, label/conf/bbox per symbol)
- Phase 8 — Reconstruction — COMPLETE (×→*, ÷→/, ordering preserved)
- Phase 9 — Parser — COMPLETE (no `eval()` — tested; validation + graceful errors)
- Phase 10 — Solver — COMPLETE (linear/arithmetic; tested incl. div-by-zero)
- Phase 11 — Canvas — COMPLETE (white canvas, dark stroke, CLEAR/SOLVE, shared backend)
- Phase 12 — UI redesign — COMPLETE (dark theme `theme.py`, `canvas.py`, 2-col INPUT/RESULT, tabs, live confidence + inference ms, details expander, pipeline section, live examples, model panel from `metrics.json`, error states; headless boot verified)
- Phase 13 — Testing — COMPLETE (`pytest`: **19 passed**, incl. `test_examples.py` on shipped inputs)
- Phase 14 — E2E cases — COMPLETE (6/6 equations, 38/38 symbols, solutions exact; no hardcoding)
- Phase 15 — CROHME eval — COMPLETE (20 V1 scans: symbol 67.7%, equation 0/20, solution 25%; `crohme_eval.json`; root cause visually proven = stroke fragmentation; documented, not forced)
- Phase 16 — Robustness — COMPLETE (symbol/equation/solution split; ~0.2 s/eq; confidence-gated warnings)
- Phase 17 — Cleanup — COMPLETE (`__pycache__` purged, minimal pinned requirements, models ship for demo, raw data gitignored)
- Phase 18 — Documentation — COMPLETE (README: 21-section coverage incl. HASYv2/CROHME usage, results, install, usage, limitations, future work)

## Datasets (archives untouched in data/raw/)
- HASYv2: 15/16 V1 classes mapped (no `=`); 3,138 real samples; imbalance handled by seeded augmentation.
- CROHME23: expression benchmark + 21,813 InkML training symbols (5,370 x, 1,500 real `=`).

## Key engineering decisions (all verified, none mocked)
1. Train/inference normalization unified (`resize_with_padding` everywhere).
2. Anti-leakage splits (sources/files split before augmentation).
3. CROHME anchoring for scarce classes (x recall 47% → 98.7%).
4. Honest metrics: 96.48% real-test reported over 99.43% synthetic-test; CROHME failures documented with visual proof.

## Known limitations / next
- V1 vocab, single-line, single `=`; CROHME scans need stroke-grouping segmentation (spec'd future work: CNN+CTC/Transformer, extended symbols, quadratics).

## How to run
`venv\Scripts\python run.py` → http://localhost:8501 → draw `2x + 5 = 15` → SOLVE → `x = 5`.

## Fix log — 2026-10-04 — Streamlit st.image() width="stretch" TypeError
- Issue: MODEL INSPECTION crashed with `TypeError: '<=' not supported between
  instances of 'str' and 'int'` at `app/ui/pipeline_view.py:54`
  (`st.image(..., width="stretch")`).
- Cause: the pinned venv runs Streamlit 1.44.0, whose `st.image` signature is
  `width: int | None` (plus `use_container_width: bool`). The string
  `"stretch"` is only accepted by newer Streamlit (e.g. 1.60); on 1.44 it
  raises TypeError. (Global Python has 1.60, which masked the bug.)
- Fix: replaced all 4 `width="stretch"` occurrences with the 1.44-compatible
  `use_container_width=True` (no hardcoded pixel width; images fill their
  container as intended):
  `app/ui/pipeline_view.py` (3 calls: preprocessed, original, symbol crops) +
  `app/main.py` (1 call: upload preview). No CNN/preprocessing/segmentation/
  solver changes.
- UI errors: wrapped MODEL INSPECTION rendering in try/except that logs the
  full traceback via `traceback.print_exc()` (terminal/logs) and shows users
  only `st.error("Unable to display model inspection results.")`.
- Verification: `pytest` **24 passed**; live pipeline re-run on shipped
  inputs — `2x+5=15→x=5`, `3x-7=11→x=6`, `7x=35→x=5`, `4x+2=18→x=4`,
  `12+8=20`/`20÷4=5` correct; solver direct `3*x+5=32→x=9` with steps
  `Subtract 5 → 3x = 27`, `Divide by 3 → x = 9`; fake-st harness confirms
  inspection renders with `use_container_width=True` and failure path shows
  the clean message; headless `streamlit run` boots `ok` with no traceback.
  NOTE: a stale pre-fix server may still occupy :8502 — restart it to pick
  up this fix.
