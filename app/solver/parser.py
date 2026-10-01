"""Expression cleanup + validation + SymPy conversion (local only)."""

from __future__ import annotations

import re

ALLOWED_CHARS = set("0123456789+-*/x=() .")

DISPLAY_TO_SYMPY = {"\u00d7": "*", "\u00f7": "/", "X": "x"}


def normalize_expression(expr: str) -> str:
    """Map display symbols (× ÷) to SymPy ops, fix implicit multiplication."""
    s = expr.strip()
    for k, v in DISPLAY_TO_SYMPY.items():
        s = s.replace(k, v)
    s = s.replace(" ", "")
    # implicit multiplication: digit/x/')' directly before 'x' or '('
    s = re.sub(r"(?<=[0-9x)])(?=[x(])", "*", s)
    # collapse duplicate operators gracefully ('++' -> '+')
    s = re.sub(r"\+\+", "+", s)
    s = re.sub(r"--", "+", s)
    return s


def validate_expression(expr: str) -> tuple[bool, str]:
    if not expr:
        return False, "Empty expression."
    bad = set(expr) - ALLOWED_CHARS - {"*", "/"}
    # normalize first so × ÷ pass
    norm = normalize_expression(expr)
    bad = set(norm) - ALLOWED_CHARS
    if bad:
        return False, f"Unsupported characters: {sorted(bad)}"
    if norm.count("=") > 1:
        return False, "At most one '=' is supported in V1."
    if re.search(r"[*/]{2,}", norm):
        return False, "Consecutive operators are not supported."
    return True, "OK"


def split_equation(expr: str) -> tuple[str, str | None]:
    norm = normalize_expression(expr)
    if "=" in norm:
        lhs, rhs = norm.split("=", 1)
        return lhs, rhs
    return norm, None


def to_sympy(expr: str):
    """Parse a single-side expression into a SymPy object."""
    from sympy import Symbol
    from sympy.parsing.sympy_parser import parse_expr
    norm = normalize_expression(expr)
    return parse_expr(norm, local_dict={"x": Symbol("x")})
