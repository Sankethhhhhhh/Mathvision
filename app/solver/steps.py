"""Deterministic step-by-step engine for V1 equations (local, SymPy only).

Generates textbook-style algebraic steps for single-variable linear
equations and single-step evaluations for plain arithmetic. Every
before -> after transition is verified with SymPy to preserve the
equation before it is returned — never display an invalid step.

Nothing is hardcoded: all operations, numbers and equations derive
from the parsed input. Returns None when the input is outside the
supported V1 step scope ( caller shows a "not supported" message ).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Step:
    operation: str   # e.g. "Subtract 5 from both sides"
    before: str      # human-readable equation before the operation
    after: str       # human-readable equation after the operation


def _fmt_num(v) -> str:
    """Compact human-readable number: 5, 10/3, 2.5 (no trailing noise)."""
    from sympy import Float, Integer, Rational

    if isinstance(v, Integer):
        return str(int(v))
    if isinstance(v, Rational):
        return f"{int(v.p)}/{int(v.q)}"
    if isinstance(v, Float):
        return f"{float(v):g}"
    try:
        iv = int(v)
        if v == iv:
            return str(iv)
    except Exception:
        pass
    try:
        return f"{float(v):g}"
    except Exception:
        return str(v)


def _fmt_side(expr) -> str:
    """Human-readable linear side: '3x + 5', 'x - 7', '5', 'x'."""
    from sympy import Poly, Symbol

    x = Symbol("x")
    try:
        poly = Poly(expr, x)
        if poly is not None and poly.degree() is not None and poly.degree() <= 1:
            coeffs = poly.all_coeffs()
            if len(coeffs) == 2:
                a, b = coeffs
            else:  # constant
                a, b = 0, coeffs[0]
            parts: list[str] = []
            if a != 0:
                if a == 1:
                    parts.append("x")
                elif a == -1:
                    parts.append("-x")
                else:
                    parts.append(f"{_fmt_num(a)}x")
            if b != 0:
                if not parts:
                    return _fmt_num(b)
                sign = "+" if b > 0 else "-"
                parts.append(f"{sign} {_fmt_num(abs(b))}")
            return " ".join(parts) if parts else "0"
    except Exception:
        pass
    return str(expr).replace("*", "")


def _equivalent(lhs1, rhs1, lhs2, rhs2) -> bool:
    """True when both linear equations share the same solution set.

    Two equations are equivalent iff their (lhs - rhs) residuals differ
    only by a nonzero constant factor (covers both "same operation on
    both sides" steps with factor 1 and "divide both sides" steps).
    """
    from sympy import expand, simplify

    try:
        d1, d2 = expand(lhs1 - rhs1), expand(lhs2 - rhs2)
        if d2 == 0:
            return d1 == 0
        ratio = simplify(d1 / d2)
        return bool(ratio.is_number) and ratio != 0
    except Exception:
        return False


def _num(value) -> str:
    return _fmt_num(value)


def _pretty_arith(norm: str) -> str:
    """Human-readable arithmetic: '12+8' -> '12 + 8', '*' -> '×'."""
    import re

    s = norm.replace("*", " * ").replace("/", " / ")
    s = re.sub(r"([+\-])", r" \1 ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.replace("*", "×").replace("/", "÷")


def explain_linear(lhs, rhs) -> list[Step] | None:
    """Build verified steps for a linear Eq(lhs, rhs). None if out of scope."""
    from sympy import Poly, Symbol, expand

    x = Symbol("x")
    try:
        if Poly(lhs - rhs, x).degree() != 1:
            return None
    except Exception:
        return None

    steps: list[Step] = []
    cl, cr = expand(lhs), expand(rhs)

    def show(l, r) -> tuple[str, str]:
        return (_fmt_side(l), _fmt_side(r))

    # 1. Collect x-terms on the left (only when x appears on the right).
    try:
        x_on_right = Poly(cr, x).degree() == 1
    except Exception:
        x_on_right = False
    if x_on_right:
        try:
            pr = Poly(cr, x)
            xpart = pr.nth(1) * x  # the variable term on the right
        except Exception:
            return None
        nl, nr = expand(cl - xpart), expand(cr - xpart)
        if not _equivalent(cl, cr, nl, nr):
            return None
        b, a = show(cl, cr), show(nl, nr)
        term = _fmt_side(xpart)
        steps.append(Step(f"Subtract {term} from both sides",
                          f"{b[0]} = {b[1]}", f"{a[0]} = {a[1]}"))
        cl, cr = nl, nr

    # 2. Move the additive constant off the left side.
    try:
        pl = Poly(cl, x)
        const = pl.nth(0)
    except Exception:
        return None
    if const != 0:
        nl, nr = expand(cl - const), expand(cr - const)
        if not _equivalent(cl, cr, nl, nr):
            return None
        b, a = show(cl, cr), show(nl, nr)
        op = (f"Subtract {_num(const)} from both sides" if const > 0
              else f"Add {_num(-const)} to both sides")
        steps.append(Step(op, f"{b[0]} = {b[1]}", f"{a[0]} = {a[1]}"))
        cl, cr = nl, nr

    # 3. Normalize the coefficient of x to 1.
    try:
        coeff = Poly(cl, x).nth(1)
    except Exception:
        return None
    if coeff != 1:
        if coeff == 0:
            return None
        nl, nr = expand(cl / coeff), expand(cr / coeff)
        if not _equivalent(cl, cr, nl, nr):
            return None
        b, a = show(cl, cr), show(nl, nr)
        steps.append(Step(f"Divide both sides by {_num(coeff)}",
                          f"{b[0]} = {b[1]}", f"{a[0]} = {a[1]}"))
        cl, cr = nl, nr

    # 4. Final state must be exactly `x = solution`.
    try:
        if expand(cl - x) != 0:
            return None
    except Exception:
        return None
    return steps


def explain_expression(norm_lhs: str, norm_rhs: str | None) -> list[Step] | None:
    """Entry point: normalized strings -> verified steps, or None.

    Covers linear equations with x, arithmetic checks (12+8=20) and
    bare values (12+8). Anything else (non-linear, errors) -> None.
    """
    from sympy import Eq, sympify

    try:
        if norm_rhs is not None:
            lhs, rhs = sympify(norm_lhs), sympify(norm_rhs)
            if "x" in (norm_lhs + norm_rhs):
                steps = explain_linear(lhs, rhs)
                if not steps:
                    return None
                # confirm the derived answer matches SymPy's own solve()
                from sympy import Symbol, solve

                sol = solve(Eq(lhs, rhs), Symbol("x"))
                final = steps[-1].after.split("=", 1)[1].strip()
                if sol and final != _fmt_side(sol[0]):
                    return None
                return steps
            # arithmetic check: one evaluation step, numerically verified
            lv, rv = float(lhs.evalf()), float(rhs.evalf())
            pretty = _pretty_arith(norm_lhs)
            if abs(lv - rv) < 1e-9:
                return [Step("Evaluate the expression",
                             pretty, f"= {_fmt_num(lhs.evalf())}")]
            return None
        val = sympify(norm_lhs)
        if "x" in norm_lhs:
            return None
        pretty = _pretty_arith(norm_lhs)
        return [Step("Evaluate the expression", pretty, f"= {_fmt_num(val.evalf())}")]
    except Exception:
        return None
