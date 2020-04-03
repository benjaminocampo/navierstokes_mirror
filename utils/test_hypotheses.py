import statsmodels.api as sm

def test_hypotheses(mnts_hyp1, mnts_hyp2):
    mu0 = mnts_hyp1.mean()
    test_statistic, p_value = sm.stats.ztest(
        mnts_hyp2, 
        value = mu0, 
        alternative = 'smaller')
    return (p_value < 0.01)
         