"""Step-by-step engine tests — verified algebraic derivations, no hardcoding."""
from __future__ import annotations

import re

from sympy import Eq, Symbol, simplify, sympify

from app.solver.equation_solver import solve_expression


def _parse(side: str):
    """Parse human display form ('3x + 5') back to SymPy."""
    s = re.sub(r"(?<=[0-9)])(?=[x(])", "*", side.replace(" ", ""))
    return sympify(s)


def _check_chain(expr: str, expected_final: str, n_steps: int):
    r = solve_expression(expr)
    assert r.kind == "linear_solution", r.message
    steps = r.data.get("steps", [])
    assert len(steps) == n_steps, steps
    # chain continuity: each after == next before
    for a, b in zip(steps, steps[1:]):
        assert a["after"] == b["before"]
    # first before is the input equation (normalized spacing aside)
    assert steps[0]["before"].replace(" ", "") == expr.replace(" ", "")
    # final answer matches solver message
    assert steps[-1]["after"].split("=", 1)[1].strip() == expected_final
    assert expected_final in r.message
    # every transition preserves the solution set
    x = Symbol("x")
    b0 = steps[0]["before"].split("=", 1)
    l0, r0 = _parse(b0[0]), _parse(b0[1])
    for s in steps:
        b1 = s["after"].split("=", 1)
        l1, r1 = _parse(b1[0]), _parse(b1[1])
        assert simplify((l0 - r0) / (l1 - r1)).is_number
        l0, r0 = l1, r1
    assert solve_check(expr, expected_final)


def solve_check(expr: str, expected_final: str) -> bool:
    lhs_s, rhs_s = expr.split("=", 1)
    sol = __import__("sympy").solve(
        Eq(_parse(lhs_s), _parse(rhs_s)), Symbol("x"))
    return bool(sol) and str(sol[0]) == expected_final


def test_steps_addition_form():
    _check_chain("3x+5=32", "9", 2)


def test_steps_spec_examples():
    _check_chain("2x+5=15", "5", 2)
    _check_chain("3x-7=11", "6", 2)
    _check_chain("7x=35", "5", 1)
    _check_chain("4x+2=18", "4", 2)
    _check_chain("5x-10=20", "6", 2)


def test_steps_variable_both_sides():
    _check_chain("2x+5=x+10", "5", 2)


def test_steps_arithmetic():
    r = solve_expression("12+8=20")
    assert r.kind == "arithmetic_check"
    assert len(r.data.get("steps", [])) == 1


def test_steps_unsupported_returns_none():
    r = solve_expression("x*x=4")  # non-linear: solution but no steps
    assert r.kind == "linear_solution"
    assert r.data.get("steps", []) == []
    r = solve_expression("2++")
    assert r.kind == "error"
