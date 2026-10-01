"""Phase 0: verify every module imports cleanly (no model weights needed)."""

import importlib


def test_imports():
    for mod in [
        "app",
        "app.preprocessing.image_processor",
        "app.segmentation.symbol_segmenter",
        "app.recognition.cnn_model",
        "app.recognition.predictor",
        "app.solver.parser",
        "app.solver.equation_solver",
        "app.ui.components",
        "training.augmentation",
    ]:
        importlib.import_module(mod)
