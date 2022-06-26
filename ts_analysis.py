# This script contains tools used to analyse time series.
# This also has tools to analyse the whole dataset used in regression modelling.
import statsmodels.tsa as smtsa
import numpy as np
import statsmodels.formula.api as smf  # statistical analysis
import statsmodels.api as sm           # statistical analysis
from scipy import stats                # statistical analysis
from typing import Sequence, Dict, Callable
from pprint import pprint

tests = {
    'stationarity tests': {
        'ADF test': smtsa.stattools.adfuller,  # H0: exsistence of unit root
        'KPSS test': smtsa.stattools.kpss,     # H0: stationarity test (level or trend)
        'RUR test': smtsa.stattools.range_unit_root_test # H0: stationarity test
    },
    'normality tests': {
        "D'Agostino's K-square test" : stats.normaltest,
        "Shapiro-Wilk test"          : stats.shapiro,
        "Kolmogorov-Smirnov test"    : stats.kstest,
        "Lilliefors' test"           : sm.stats.diagnostic.lilliefors
    },

    # cointegration?

    # AR detection
}

#
# this function applies multiple tests defined in the tests dictionary to a dataset
def normality_tests(series: Sequence,
                    tests = tests['normality tests'],
                    alpha = 0.05) -> Dict[str, str]:
    # test results are stored in a dictionary
    results: Dict[str, str] = {}

    # loop through the tests given by the tests dictionary
    for test in tests:
        # if there is any error, it will be catched such that other tests won't be jeopardised
        try:
            # the K-S test needs an additional argument
            if test == "Kolmogorov-Smirnov test":
                # call the test and get the p-value
                p_value = tests[test](series, cdf = stats.norm.cdf)[1]
            else:
                p_value = tests[test](series)[1]
        
        # if an error occors,
        #   store error message in the corresponding position of results,
        #   and move to the next iteration
        except Exception as error:
            results[test] = error                      # store the error message in results dictionary
            print(f'{test} has failed since {error}')  # show that there is an error in the console
            continue                                   # proceed to the next iteration

        # determine whether to reject the null hypothesis or not depending on the p-value and the alpha level
        results[test] = 'Not normal' if p_value < alpha else 'Fail to reject normality'

    return results

def stationarity_tests(series: Sequence,
                       tests = tests['stationarity tests'],
                       alpha = 0.05) -> Dict[str, str]:
    # test results are stored in a dictionary
    results: Dict[str, str] = {}

    # loop through the tests provided
    for test in tests:
        try:
            if test == 'RUR test':
                p_value = tests[test](series)[1]
            else:
                p_value = tests[test](series, regression='ct')[1]
        except Exception as error:
            results[test] = error
            print(f'{test} has failed since {error}.')
            continue
        
        if test == 'ADF test':
            results[test] = 'No unit root, can be stationary' if p_value < alpha else 'Can be unit root process/non-stationary'
        elif test == 'KPSS test':
            results[test] = 'Not level nor trend stationary' if p_value < alpha else 'Can be level or trend stationary'
        elif test == 'RUR test':
            results[test] = 'Not stationary' if p_value < alpha else 'Can be stationary'
    
    return results


if __name__ == '__main__':
    x = np.random.normal(loc=0, scale=1, size=3000)
    
    # normality tests
    pprint(normality_tests(x))

    # stationarity tests
    pprint(stationarity_tests(x))
