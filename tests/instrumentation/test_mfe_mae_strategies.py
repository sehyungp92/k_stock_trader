"""Verify all strategies wire mfe_mae_context to on_exit_fill."""
import inspect


def test_kpr_wires_mfe_mae():
    from strategy_kpr import main
    source = inspect.getsource(main)
    assert "mfe_mae_context" in source


def test_nulrimok_wires_mfe_mae():
    from strategy_nulrimok import main
    source = inspect.getsource(main)
    assert "mfe_mae_context" in source


def test_pcim_wires_mfe_mae():
    from strategy_pcim import main
    source = inspect.getsource(main)
    assert "mfe_mae_context" in source
