"""Tests for KMP MFE/MAE state tracking and context building."""
import math
from strategy_kmp.core.state import SymbolState
from strategy_kmp.core.exits import update_trail, build_mfe_mae_context


class TestMinAdverseTracking:
    def test_symbol_state_has_min_adverse(self):
        s = SymbolState(code="005930")
        assert hasattr(s, "min_adverse")
        assert s.min_adverse == math.inf

    def test_update_trail_tracks_min_adverse(self):
        s = SymbolState(code="005930")
        s.entry_px = 70000
        s.entry_ts = 1000000000.0
        s.structure_stop = 69000
        s.max_fav = 70000
        s.min_adverse = 70000

        update_trail(s, 69500, "mixed")
        assert s.min_adverse == 69500

        update_trail(s, 71000, "mixed")
        assert s.min_adverse == 69500  # Should not increase

        update_trail(s, 69200, "mixed")
        assert s.min_adverse == 69200

    def test_max_fav_still_tracks(self):
        s = SymbolState(code="005930")
        s.entry_px = 70000
        s.entry_ts = 1000000000.0
        s.structure_stop = 69000
        s.max_fav = 70000
        s.min_adverse = 70000

        update_trail(s, 73000, "mixed")
        assert s.max_fav == 73000


class TestBuildMfeMaeContext:
    def test_normal_trade(self):
        s = SymbolState(code="005930")
        s.entry_px = 70000
        s.structure_stop = 69000
        s.max_fav = 73000
        s.min_adverse = 69500

        ctx = build_mfe_mae_context(s)
        assert ctx["mfe_price"] == 73000
        assert ctx["mae_price"] == 69500
        assert abs(ctx["mfe_r"] - 3.0) < 0.01
        assert abs(ctx["mae_r"] - 0.5) < 0.01

    def test_no_adverse(self):
        s = SymbolState(code="005930")
        s.entry_px = 70000
        s.structure_stop = 69000
        s.max_fav = 71000
        s.min_adverse = math.inf

        ctx = build_mfe_mae_context(s)
        assert ctx["mfe_price"] == 71000
        assert ctx["mae_price"] is None


class TestMainWiring:
    def test_on_exit_fill_has_mfe_mae_context(self):
        """Verify main.py passes mfe_mae_context to on_exit_fill."""
        import inspect
        from strategy_kmp import main
        source = inspect.getsource(main)
        assert "mfe_mae_context" in source
