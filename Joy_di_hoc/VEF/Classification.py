
import time
import pandas as pd
from scipy.stats import mode, zscore
import numpy as np
from scipy.stats import mode, zscore
import matplotlib.pyplot as plt
import seaborn
import re

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split


def get_churn_data(path):
    churn = pd.read_excel(CHURN)
    churn.drop(columns='Phone', inplace=True)

    # Process columns name
    new_columns = []
    for i in churn.columns:
        new_name = re.sub(r'\W+', '_', i)
        if re.match('^\d', new_name):
            new_name = '_' + new_name
        new_columns.append(new_name)
    churn.columns = new_columns
    churn = pd.get_dummies(churn)
    return churn


if __name__ == "__main__":
    start_time = time.time()
    pd.set_option("display.max_rows", None, "display.max_columns", 50, 'display.width', 1000)
    CHURN = "https://github.com/pnhuy/datasets/raw/master/Churn.xls"
    df = get_churn_data(CHURN)
    lower_names = [name.lower() for name in df.columns]
    df.columns = lower_names

    # feature selection
    X = df.drop(columns=['churn'])
    Y = df.churn
    names = X.columns
    rf = RandomForestClassifier()
    rf.fit(X, Y)

    feature_importance = pd.DataFrame(
        {
            'names': names,
            'feature_importance': rf.feature_importances_
        }
    )

    sorted_fi = feature_importance.sort_values(by="feature_importance", ascending=False)
    top_feature = sorted_fi['names'].head(10).values.tolist()

    subset_churn = df[top_feature + ['churn']]
    # Chia tâpj dữ liệu thành train và test. tuy nhiên dữ liệu bị lấy lần lượt, không đảm bảo tính random => nên sử dụng thư viện để chia tâp dữ liệu
    # churn_train = subset_churn[:2222]
    # churn_test = subset_churn[2222:]

    churn_train, churn_test = train_test_split(subset_churn, test_size=0.3)
    formular = 'churn ~ ' + ' + '.join(top_feature)
    print(formular)

    # logreg = smf.logit(formular, data=churn_train).fit()

    # Loại bỏ các biến có p_value cao:
    k = 'churn ~ custserv_calls + int_l_plan + intl_calls + night_mins'
    logreg = smf.logit(k, data=churn_train).fit()
    print(logreg.summary())



    # OLS base top feature

    # df = df[top_feature + ['saleprice']]
    # ols_features = ""
    # for feature in top_feature:
    #     ols_features = ols_features + ' + ' + feature
    # ols_features = ols_features[3:]
    # print(ols_features + "\n")
    #
    # results = smf.ols(f"saleprice ~ {ols_features}", data=df).fit()
    # print(results.summary())
