from scipy import stats
import numpy as np
import pandas as pd

import statsmodels.api as sm
import pylab
import time
import seaborn as sns
import matplotlib.pyplot as plt


def normality_test(x: list):
    '''
    doc:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html
    https://nghiencuugiaoduc.com.vn/bai-5-phan-phoi-chuan-normal-distribution/
       n > 300: |skewness| <2 and |kurtosis| < 7: normal distribution
       n < 300: - if n < 50: Shapiro-Wilk test
                  else n > 50: Kolmogorov-smirnov (Ks test)
                - Ho: Normal distribution
                  H1: Not normal distribution
    '''
    # distplot
    sns.distplot(x=x, hist_kws=dict(edgecolor="w", linewidth=1), bins=25, color="r")
    plt.show()

    # skewness and kurtosis
    print("Skewness: %f", stats.skew(x))
    print("Kurtosis: %f", stats.kurtosis(x))

    # Quantile-Quantile Plot
    # doc: https://towardsdatascience.com/significance-of-q-q-plots-6f0c6e31c626
    #      https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html

    stats.probplot(x=x, dist="norm", plot=pylab)
    pylab.show()


    # Shapiro-Wilk
    shapiro_test = stats.shapiro(x)
    print("Shapiro-Wilk test: Critical value: ", shapiro_test.statistic, ", p_value: ", shapiro_test.pvalue)
    # Kolmogorov-smirnov
    ks_test = stats.kstest(x, 'norm', N=100)
    print("Kolmogorov-smirnov test: Critical value: ", ks_test.statistic, ", p_value: ", ks_test.pvalue)


def grubbs_test(x): # outlier test
    '''
        H0: There are no outliers.
        H1: There is at least one outlier.
    '''
    n = len(x)
    mean_x = np.mean(x)
    sd_x = np.std(x)
    numerator = max(abs(x - mean_x))
    g_calculated = numerator / sd_x
    print("Grubbs Calculated Value:", g_calculated)
    t_value = stats.t.ppf(1 - 0.05 / (2 * n), n - 2)
    g_critical = ((n - 1) * np.sqrt(np.square(t_value))) / (np.sqrt(n) * np.sqrt(n - 2 + np.square(t_value)))
    print("Grubbs Critical Value:", g_critical)
    if g_critical > g_calculated:
        print(
            "From grubbs_test we observe that calculated value is lesser than critical value, Accept null hypothesis and conclude that there is no outliers\n")
    else:
        print(
            "From grubbs_test we observe that calculated value is greater than critical value, Reject null hypothesis and conclude that there is an outliers\n")


if __name__ == "__main__":
    start_time = time.time()
    pd.set_option("display.max_rows", None, "display.max_columns", 30, 'display.width', 500)
    # url = "https://raw.githubusercontent.com/tiwari91/Housing-Prices/master/train.csv"
    # df = pd.read_csv(url, sep=",")
    # x = df['SalePrice']
    # sns.distplot(x, hist_kws=dict(edgecolor="w", linewidth=1), bins=25, color="r")
    # plt.title("Sale_price distribution")
    # normality_test(x=x)


    test = np.random.normal(0, 1, 1000)
    print(test)

    # sm.qqplot(test, line='45')
    # py.show()

    print("\n --- total time to process %s seconds ---" % (time.time() - start_time))
