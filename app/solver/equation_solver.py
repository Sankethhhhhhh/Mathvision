"""Local equation solver built on SymPy (no network calls)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SolveResult:
    kind: str               # 'linear_solution' | 'arithmetic_check' | 'value' | 'error'
    expression: str         # normalized expression
    message: str            # human-readable summary
    data: dict              # machine-readable details


def solve_expression(expr: str) -> SolveResult:
    from sympy import Eq, Symbol, solve, sympify

    from app.solver.parser import normalize_expression, split_equation, validate_expression

    ok, reason = validate_expression(expr)
    if not ok:
        return SolveResult("error", expr, reason, {})

    norm = normalize_expression(expr)
    lhs_s, rhs_s = split_equation(norm)
    x = Symbol("x")

    try:
        if rhs_s is not None:
            lhs, rhs = sympify(lhs_s), sympify(rhs_s)
            if "x" in norm:
                sol = solve(Eq(lhs, rhs), x)
                if not sol:
                    return SolveResult("error", norm,
                                       "No solution found.", {"lhs": str(lhs), "rhs": str(rhs)})
                return SolveResult(
                    "linear_solution", norm,
                    f"{' , '.join(f'x = {s}' for s in sol)}",
                    {"solutions": [str(s) for s in sol],
                     "lhs": str(lhs), "rhs": str(rhs)})
            lv, rv = float(lhs.evalf()), float(rhs.evalf())
            correct = abs(lv - rv) < 1e-9
            return SolveResult(
                "arithmetic_check", norm,
                f"{lv:g} = {rv:g} → {'correct ✔' if correct else 'incorrect ✘'}",
                {"lhs_value": lv, "rhs_value": rv, "correct": correct})
        val = sympify(lhs_s)
        if "x" in norm:
            sol = solve(val, x)
            return SolveResult("linear_solution", norm,
                               f"{' , '.join(f'x = {s}' for s in sol)}",
                               {"solutions": [str(s) for s in sol]})
        return SolveResult("value", norm, f"= {float(val.evalf()):g}",
                           {"value": float(val.evalf())})
    except Exception as e:  # noqa: BLE001 — surface as user-facing error
        return SolveResult("error", norm, f"Could not solve: {e}", {})
