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
