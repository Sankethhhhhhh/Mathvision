"""MathVision solver package (local SymPy engine, no external APIs)."""
from app.solver.equation_solver import SolveResult, solve_expression
from app.solver.parser import normalize_expression, split_equation, validate_expression

__all__ = [
    "SolveResult",
    "solve_expression",
    "normalize_expression",
    "split_equation",
    "validate_expression",
]
