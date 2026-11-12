"""Run inherited lifecycle and new risk/selection tests against frozen modules."""
from scripts import olr_precision_test_v3 as runner
from scripts import olr_repair_study_v4 as t
TESTS=(*runner.TESTS,'tests/strategy_olr/test_risk_repair_v4.py')
def main():
    runner.t=t;runner.TESTS=TESTS;runner.main()
if __name__=='__main__':main()
