# MathVision — Handwritten Mathematical Expression Recognition and Solver Using CNN

> **Academic Mini-Project** | Neural Networks and Deep Learning (NNDL)

MathVision is a locally-run end-to-end deep learning system that recognizes handwritten mathematical equations from interactive canvas drawings or uploaded images, reconstructs the expressions using spatial symbol segmentation, and solves them locally using symbolic computation.

---

## 🎯 Project Objective

The core goal of MathVision is to build a fully local, self-contained neural network system without relying on cloud APIs or external AI services (no OpenAI, Gemini, Claude, or cloud OCR APIs). 

### Fundamental Pipeline
```text
User Input (Canvas / Image)
           ↓
   Image Preprocessing (Grayscale, Noise Removal, Binarization)
           ↓
   Symbol Segmentation (Contour Detection, Spatial Sorting, Bounding Boxes)
           ↓
   CNN Classification (Custom 28x28 Handwritten Symbol Classifier)
           ↓
   Equation Reconstruction (Visual Symbol to Math Operator Mapping)
           ↓
   Mathematical Parsing & Solving (Local SymPy Parsing)
           ↓
   Visual Output & Pipeline Explainability (Streamlit UI)
```

---

## 🔢 Supported Symbol Vocabulary (Version 1)

* **Digits (0-9):** `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`
* **Operators:** `+`, `-`, `×` (mapped to `*`), `÷` (mapped to `/`)
* **Equals:** `=`
* **Variable:** `x`

### Target Equation Examples (V1)
* `2x + 5 = 15`
* `3x - 7 = 11`
* `4x + 2 = 18`
* `20 ÷ 4 = 5`
* `12 + 8 = 20`

*Future extensions will include parentheses `()`, powers `²`, decimal points, fractions, quadratic equations, and multiple variables (`y`, `z`).*

---

## 🧠 Neural Network Architecture

A custom Convolutional Neural Network (CNN) trained specifically on handwritten digits, mathematical operators, and variable characters.

* **Input Size:** `28 × 28 × 1` (Grayscale)
* **Architecture:** `Conv2D → ReLU → MaxPooling → Conv2D → ReLU → MaxPooling → Flatten → Dense → Dropout → Dense (Softmax)`
* **Loss Function:** Categorical Cross-Entropy
* **Optimizer:** Adam

---

## 📂 Project Structure

```text
MathVision/
│
├── app/
│   ├── __init__.py
│   ├── main.py                # Streamlit entry point
│   ├── preprocessing/         # Image binarization, noise removal & padding
│   ├── segmentation/          # Contour detection & left-to-right sorting
│   ├── recognition/           # CNN model wrapper & inference pipeline
│   ├── solver/                # Expression reconstruction & SymPy engine
│   └── ui/                    # Streamlit components & visual explainability
│
├── training/
│   ├── __init__.py
│   ├── train.py               # CNN training loop & model saving
│   ├── evaluate.py            # Confusion matrix, accuracy, loss plots & metrics
│   └── augmentation.py        # Custom data augmentation routines
│
├── data/
│   ├── raw/                   # Raw handwritten symbol dataset
│   ├── processed/             # Cleaned, normalized 28x28 samples
│   └── test/                  # Sample handwritten equation test images
│
├── models/                    # Saved CNN models (.keras) and class_names.json
├── notebooks/                 # Exploratory notebooks & data inspection
├── tests/                     # Unit tests for preprocessing, segmentation, & solver
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🗺️ 10-Phase Development Strategy

1. **Phase 1: Project Setup & Dataset Preparation** (Current)
2. **Phase 2: CNN Training for Handwritten Symbol Classification**
3. **Phase 3: Model Evaluation & Metrics Analysis**
4. **Phase 4: Handwritten Symbol Segmentation**
5. **Phase 5: Connecting Segmentation to CNN Inference**
6. **Phase 6: Equation Reconstruction Engine**
7. **Phase 7: SymPy Solver Integration**
8. **Phase 8: Streamlit Interactive UI & Pipeline Visualization**
9. **Phase 9: End-to-End Testing & Integration Verification**
10. **Phase 10: Documentation, Presentation & Viva Preparation**

---

## 🛠️ Environment Setup & Installation

### Requirements
* Python 3.11 or 3.12 (64-bit)

### Setup Virtual Environment
```bash
# Windows
py -3.12 -m venv venv
.\venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
```

---

## 📊 Results (measured, seed 42 — real-data model)

### Dataset (hybrid, stratified 1400/300/300 per class → train 22,400 / val 4,800 / test 4,800)
* **Digits 0–9:** MNIST (1500/class) + 400 synthetic print-style digits/class in train only (covers flagged `1`, barred `7`).
* **`+ − × ÷`:** real **HASYv2** strokes (`data/processed/class_mapping.json`: exact IDs 196/195/513/526), upsampled with identity-safe augmentation. Anti-leakage: sources split before augmenting.
* **`x`:** 5,370 real **CROHME InkML** symbols (file-grouped split) + HASY + font supplement — recovered held-out-writer recall from 47% to **98.7%**.
* **`=`:** 900 real **CROHME** symbols + 500 synthetic (no HASYv2 `=` class exists — verified).
* All sources normalized with the exact inference function (tight-crop + `resize_with_padding` + centre-of-mass).
* Sample grids: `assets/dataset_samples/hasy_grid.png`, `assets/dataset_samples/crohme/`.

### CNN test accuracy: **96.48%** on the real-handwriting test set, macro F1 **0.9650**
Per-class recall ≥ 91.3% (`4` weakest; `x` 98.7%, `+`/`−` 100%). The earlier 99.43% was measured on an easier synthetic-only test set — the 96.48% reflects genuine writer variation. Confusion matrix: `models/confusion_matrix.png`; curves: `models/training_history.png`; numbers: `models/metrics.json` (displayed live in the app).

### End-to-end validation (synthetic handwriting-style renders, 6/6)
| Input | Recognized | Solution |
|---|---|---|
| `2x+5=15` | `2x+5=15` | `x = 5` |
| `3x-7=11` | `3x-7=11` | `x = 6` |
| `7x=35` | `7x=35` | `x = 5` |
| `4x+2=18` | `4x+2=18` | `x = 4` |
| `12+8=20` | `12+8=20` | correct ✔ |
| `20÷4=5` | `20÷4=5` | correct ✔ |

* Symbol accuracy **38/38 (100%)**, equation accuracy **6/6**, solution accuracy **6/6**.
* Inference ≈ **0.2 s/equation** (CPU, after first-load warmup; measured live in the UI); model size **2.9 MB** (`models/mathvision_cnn.keras`).
* Tests: **19 passed** (`pytest tests`), incl. end-to-end runs on the shipped `data/test/example_*.png` inputs.

### CROHME expression-level evaluation (honest out-of-domain probe, `training/evaluate_crohme.py`)
20 single-line V1 test scans: symbol **67.7%**, equation **0/20**, solution **25%**. Root cause (visually verified): InkML renders are 1 px hairline strokes with junction gaps, so pen strokes fragment into pieces the contour segmenter cannot re-group. Mitigations shipped: adaptive morphological closing, hairline training augmentation, fragment absorption. Standing benchmark in `data/processed/crohme_eval.json`; proper fix = stroke-grouping segmentation (future work).

---

## 🚀 Usage

```bash
# 1. Real-data preparation (archives stay in data/raw, gitignored)
venv\Scripts\python -m training.prepare_hasyv2      # HASYv2 filter -> class_mapping.json
venv\Scripts\python -m training.inspect_crohme      # CROHME catalog -> crohme_info.json
venv\Scripts\python -m training.extract_crohme_symbols  # InkML symbol crops
venv\Scripts\python -m training.dataset_validation  # validation_report.json

# 2. Build hybrid set + train (saves models/mathvision_cnn.keras)
venv\Scripts\python -m training.build_hasy_dataset
venv\Scripts\python -m training.train --data data/processed/mathvision_hasy.npz --epochs 25

# 3. Evaluate (metrics.json, confusion_matrix.png) + CROHME probe
venv\Scripts\python -m training.evaluate --data data/processed/mathvision_hasy.npz
venv\Scripts\python -m training.evaluate_crohme

# 4. Run tests
venv\Scripts\python -m pytest tests -q

# 5. Launch the app (drawing canvas + image upload)
venv\Scripts\python run.py
```

> Canvas note: `streamlit-drawable-canvas` 0.9.3 is pinned with `streamlit==1.44.0`
> because newer Streamlit removed the components API it uses.

---

## ⚠️ Limitations
* V1 vocabulary only: `0-9 + - × ÷ = x`; single-line equations, at most one `=`.
* No parentheses, decimals, fractions, powers, or multi-variable equations yet.
* Segmentation assumes connected pen strokes with clear gaps; hairline vector renders (CROHME InkML scans) fragment at stroke junctions — needs stroke-grouping segmentation.
* `x` must be written distinctly from `×` (model cue: curved handwritten strokes vs straight cross).

## 🔮 Future Work
* Extended symbols (`y`, `z`, parentheses, decimal point, powers), quadratic solver path.
* Stroke-graph/sequence model (CNN + CTC/Transformer) to replace contour segmentation — closes the CROHME gap.
* Real-handwriting fine-tuning set collected from the canvas UI.

---

## 📜 License & Academic Integrity
This project is developed strictly for educational and academic evaluation purposes in the **Neural Networks and Deep Learning (NNDL)** course. All neural network models and preprocessing algorithms are trained and run locally.
