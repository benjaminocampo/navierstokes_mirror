import statsmodels.api as sm

def test_hypotheses(mnts_hyp1, mnts_hyp2):
    mu0 = mnts_hyp1.mean()
    observed_value = mnts_hyp2.mean()
    test_statistic, p_value = sm.stats.ztest(
        mnts_hyp2, 
        value = mu0, 
        alternative = 'smaller')
    print(mnts_hyp2)
    print('Observed Value: ', observed_value)
    print('Mu0: ', mu0)
    print('Test Statistic: ', test_statistic)
    print('P-Value: ', p_value)
         