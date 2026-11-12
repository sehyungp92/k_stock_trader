from scripts import olr_precision_test_v3 as runner
from scripts import olr_participation_study_v5 as t
TESTS=(*runner.TESTS,'tests/strategy_olr/test_risk_repair_v4.py','tests/strategy_olr/test_participation_v5.py')
def main():
    runner.t=t;runner.TESTS=TESTS;runner.main()
if __name__=='__main__': main()
