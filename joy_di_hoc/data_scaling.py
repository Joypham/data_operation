from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
from joy_di_hoc.data_cleansing import re_format_file, read_data
import statsmodels.api as sm
import pylab
import scipy.stats as stats


from joy_di_hoc.hypothesis_testing import normality_test
import seaborn as sns
import matplotlib.pyplot as plt


class Standardization(BaseEstimator, TransformerMixin):

    # standard_scaler = zscore(unscale_input)

    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        # scaler is nothing but a Standard Scaler object
        self.scaler = StandardScaler(copy, with_mean, with_std)
        # with some columns 'twist'
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    # the fit method, which, again based on StandardScale

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self

    # the transform method which does the actual scaling

    def transform(self, X, y=None, copy=None):
        # record the initial order of the columns
        init_col_order = X.columns

        # scale all features that you chose when creating the instance of the class
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)

        # declare a variable containing all information that was not scaled
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]

        # return a data frame which contains all scaled features and all 'not scaled' features
        # use the original order (that you recorded in the beginning)
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


'''
    need to write code for this
'''


class Normalization(BaseEstimator, TransformerMixin):
    def __init__(self, data: object, copy=True, with_mean=True, with_std=True):
        self.raw_data = data
        self.normalized_data = None
        self.inverse = None

    def minmax_scale(self):
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(self.raw_data.to_frame())
        self.normalized_data = normalized_data.flatten()
        # print(self.normalized_data)
        self.inverse = scaler.inverse_transform(normalized_data).flatten()
        # print(self.inverse)
        return self

    def log_scale(self):
        self.normalized_data = np.log(self.raw_data)
        self.inverse = np.expm1(self.normalized_data)
        return self


if __name__ == "__main__":
    pd.set_option("display.max_rows", None, "display.max_columns", 30, 'display.width', 500)
    # url = "https://raw.githubusercontent.com/tiwari91/Housing-Prices/master/train.csv"
    # df = read_data(path_name=url)
    # df = re_format_file(df=df)
    # joy = df['saleprice']
    # print(joy)

    # sns.distplot(x, hist_kws=dict(edgecolor="w", linewidth=1), bins=25, color="r")
    # plt.title("Sale_price distribution")
    # plt.show()

    # k = Normalization(data=joy).log_scale().normalized_data
    # j = Normalization(data=joy).minmax_scale().inverse
    # print(j)

    # normality_test(x=saleprice_normalized)

    # test = np.random.normal(0, 1, 1000)
    # measurements = np.random.normal(loc=20, scale=5, size=100)

    animals = ['cat', 'dog', 'guinea pig']
    animals.remove('dog')
    print(animals)






    # sns.distplot(normalized, hist_kws=dict(edgecolor="w", linewidth=1), bins=25, color="r")
    # plt.title("Sale_price distribution")
    # plt.show()
    # X_train = scaler.transform(X_train)









