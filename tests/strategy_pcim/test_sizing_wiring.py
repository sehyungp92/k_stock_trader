"""Verify that PCIM's on_entry_fill() call includes sizing_context parameter."""

import ast
import pathlib
import textwrap


MAIN_PY = pathlib.Path(__file__).resolve().parents[2] / "strategy_pcim" / "main.py"


def _get_on_entry_fill_calls(source: str):
    """Return all ast.Call nodes where the function is `instr.on_entry_fill`."""
    tree = ast.parse(source)
    results = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "on_entry_fill"
            and isinstance(func.value, ast.Name)
            and func.value.id == "instr"
        ):
            results.append(node)
    return results


def test_on_entry_fill_has_sizing_context():
    """The instr.on_entry_fill() call in main.py must pass sizing_context."""
    source = MAIN_PY.read_text(encoding="utf-8")
    calls = _get_on_entry_fill_calls(source)
    assert calls, "No instr.on_entry_fill() call found in strategy_pcim/main.py"

    for call in calls:
        kw_names = [kw.arg for kw in call.keywords]
        assert "sizing_context" in kw_names, (
            f"instr.on_entry_fill() at line {call.lineno} is missing sizing_context= keyword argument"
        )


def test_build_sizing_context_imported():
    """build_sizing_context must be imported in main.py."""
    source = MAIN_PY.read_text(encoding="utf-8")
    assert "build_sizing_context" in source, (
        "build_sizing_context is not imported in strategy_pcim/main.py"
    )


def test_sizing_constants_imported():
    """SIZING constants must be imported in main.py for stop_distance computation."""
    source = MAIN_PY.read_text(encoding="utf-8")
    assert "SIZING" in source, (
        "SIZING constants not imported in strategy_pcim/main.py"
    )
