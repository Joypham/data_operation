from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


class CustomScaler(BaseEstimator, TransformerMixin):

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