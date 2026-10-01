"""Phase 12/13: error handling + parser safety (never crash on bad input)."""
import numpy as np

from app.main import run_pipeline
from app.solver.equation_solver import solve_expression
from app.solver.parser import normalize_expression


def test_blank_image_gives_no_crops():
    white = np.ones((120, 300, 3), dtype=np.uint8) * 255
    binary, crops, details, expr, result = run_pipeline(white)
    assert crops == []
    assert expr == ""
    assert result is None


def test_empty_expression_error():
    r = solve_expression("")
    assert r.kind == "error"


def test_multiple_equals_error():
    r = solve_expression("2x+5=15=3")
    assert r.kind == "error"


def test_unsupported_chars_error():
    r = solve_expression("2y+5=15")
    assert r.kind == "error"


def test_division_by_zero_no_crash():
    r = solve_expression("5/0=5")
    assert r.kind in ("error", "arithmetic_check")


def test_malformed_expression_no_crash():
    for bad in ["+", "=", "x+", "2**3=8", "5//2=2"]:
        r = solve_expression(bad)
        assert r.kind in ("error", "value", "linear_solution", "arithmetic_check")


def test_implicit_multiplication():
    assert normalize_expression("2x+5=15") == "2*x+5=15"


def test_no_arbitrary_eval():
    import pathlib
    for f in ["app/solver/parser.py", "app/solver/equation_solver.py"]:
        src = pathlib.Path(f).read_text(encoding="utf-8")
        assert "eval(" not in src.replace("evalf", ""), f"raw eval() in {f}"
