# MathVision — Handwritten Mathematical Expression Recognition

> **Academic Mini-Project** | Neural Networks and Deep Learning (NNDL)

MathVision is an end-to-end deep learning system that recognizes handwritten mathematical equations from interactive canvas drawings or uploaded images, reconstructs the expressions using spatial symbol segmentation, and solves them locally using symbolic computation.

## Architecture

The project has been refactored into a decoupled, production-ready architecture:

- **Frontend (Vercel):** A Next.js (React + Tailwind CSS) application providing a premium, interactive user interface. It captures drawing input from an HTML5 canvas and visualizes the math pipeline.
- **Backend API:** A Python FastAPI service hosting the local MathVision CNN inference pipeline and SymPy solver.

> **Crucially:** CNN inference is local to the deployed backend. No external AI API (like OpenAI or Gemini) is used.

## Features

- **Draw & Upload:** Draw an equation natively in the browser or upload an image.
- **Local Symbol Recognition:** A custom CNN predicts each segmented handwritten symbol.
- **Math Solver:** SymPy performs symbolic solving and algebraic step generation.
- **Visual Pipeline:** UI presents the preprocessing, recognition confidence, and solve path step-by-step.

## ML Pipeline

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
JSON API Response → Next.js Visual Output

## Tech Stack

- **Frontend:** Next.js, React, Tailwind CSS, TypeScript
- **Backend:** Python, FastAPI, Uvicorn
- **Machine Learning:** TensorFlow/Keras CNN (Local weights, ~3 MB)
- **Computer Vision:** OpenCV, NumPy, Pillow
- **Math Engine:** SymPy

## Local Development

### 1. Run the Backend API
```bash
# Activate your virtual environment
pip install -r backend/requirements.txt
# Start the FastAPI server
uvicorn backend.api:app --reload --port 8000
```

### 2. Run the Frontend UI
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000` to interact with MathVision.

## Vercel Deployment

The frontend is ready to be deployed to Vercel:
1. Connect this repository to Vercel.
2. Set the **Framework Preset** to Next.js.
3. Set the **Root Directory** to `frontend`.
4. Add the Environment Variable `NEXT_PUBLIC_API_URL` pointing to your deployed FastAPI backend (e.g. `https://mathvision-api.onrender.com/api`).
5. Deploy.

## Backend Deployment

Due to Vercel Serverless Function size limits (250MB uncompressed) which TensorFlow exceeds, the backend must be deployed to a container or VM provider (e.g. Render, Railway, Fly.io):
1. Point your host to the repository.
2. Set the build command to `pip install -r backend/requirements.txt`.
3. Set the start command to `uvicorn backend.api:app --host 0.0.0.0 --port $PORT`.

## API Endpoints

- `GET /api/health` — Check if the ML model is loaded and backend is online.
- `GET /api/model-info` — Returns CNN metrics and classes.
- `POST /api/solve` — Submit an image for end-to-end inference and solving.

## Model

The CNN model size is 2.9 MB (`models/mathvision_cnn.keras`).
It supports classes: `0-9`, `+`, `-`, `×`, `÷`, `=`, `x`.
The inference logic runs entirely offline and local to the backend server.

## Example

Testing `2x + 5 = 15` yields a JSON response outlining the equation `2x + 5 = 15` and the SymPy-solved root `x = 5`, along with confidence scores for each recognized symbol.

## Limitations

- V1 vocabulary only handles single-line equations and a single variable `x`.
- Segmentation relies on clear, connected pen strokes.
- Backend deployment requires a host with >500MB memory due to TensorFlow's footprint.
