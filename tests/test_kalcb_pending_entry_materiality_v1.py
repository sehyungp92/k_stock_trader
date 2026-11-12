import importlib.util
from pathlib import Path
import unittest

p = Path(__file__).resolve().parents[1]/'backtests/strategies/kalcb/pending_entry_materiality_v1.py'
s = importlib.util.spec_from_file_location('materiality', p)
h = importlib.util.module_from_spec(s); s.loader.exec_module(h)


def row():
    return dict(completion_at='2026-01-02T09:35:00+09:00', order_id='a',
        symbol='X', day='2026-01-02', position_id='X/1', category='price_not_confirmed',
        price_confirmed=False, parent_qty_bought_before35=2, planned_qty=5,
        parent_sell_intent_at_or_before35=False, parent_sale_before35=False,
        original_reference=100, parent_entry_cost_krw=490, parent_net_pnl_krw=50)


def fills():
    return [dict(order_id='a', side='BUY', qty=q, price=p,
        timestamp=t, metadata=dict(execution_id=i, fee=1)) for q,p,t,i in [
        (2,100,'2026-01-02T09:30:00+09:00','one'),
        (3,97,'2026-01-02T09:35:00+09:00','two')]]


class MaterialityTests(unittest.TestCase):
    def test_same_clock_fill_is_future(self):
        m=h.measure(row(),fills());self.assertEqual(m['later_qty'],3)
        self.assertEqual(m['later_price_minus_original_reference_krw'],-9)
        self.assertIsNone(m['candidate_pnl'])

    def test_future_outcomes_do_not_select_potential(self):
        r=row();a=h.measure(r,fills());r['parent_net_pnl_krw']=-999
        b=h.measure(r,fills()[:1]);self.assertEqual(a['potential_remainder_observation'],b['potential_remainder_observation'])
        self.assertFalse(b['subsequent_fill_observed']);self.assertFalse(b['pending_status_at35_qualified'])

    def test_original_exit_blocks(self):
        r=row();r['parent_sell_intent_at_or_before35']=True
        self.assertFalse(h.measure(r,fills())['potential_remainder_observation'])

    def test_fully_filled_not_pending(self):
        r=row();r['planned_qty']=2
        self.assertFalse(h.measure(r,fills()[:1])['potential_remainder_observation'])

    def test_duplicate_fill_rejected(self):
        with self.assertRaises(ValueError):h.measure(row(),fills()+fills())

    def test_duplicate_lifetime_rejected(self):
        m=h.measure(row(),fills())
        with self.assertRaises(ValueError):h.summary([m,m])

    def test_unsupported_clock_retained_without_classification(self):
        r=row();r['category']='unsupported_opening_clock'
        self.assertFalse(h.measure(r,fills())['potential_remainder_observation'])

    def test_overfill_rejected(self):
        r=row();r['planned_qty']=4
        with self.assertRaises(ValueError):h.measure(r,fills())


if __name__=='__main__':unittest.main()
