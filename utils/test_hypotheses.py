import statsmodels.api as sm


def test_hypotheses(sample_hyp1, sample_hyp2):
    """
    Use a ztest procedure to decide if the observed value
    given by sample_hyp2 rejects the null hypothesis.

    Parameters
    -------
    sample_hyp1: Panda Series object
        sample of null hypothesis.
    mnts_hyp2: Panda Series object
        sample of the observed value.

    Returns
    -------
    bool:
        True, if the null hypothesis is rejected.
        False, if there is no strong evidence to reject it.
    """

    mu0 = sample_hyp1.mean()
    test_statistic, p_value = sm.stats.ztest(
        sample_hyp2,
        value=mu0,
        alternative='smaller')
    return (p_value < 0.01)
