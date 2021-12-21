import numpy as np
import pandas as pd
from scipy.stats import mode, zscore
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

import re
import time
from joy_di_hoc.data_cleansing import *
import pickle


class data_modeling():
    def __init__(self, data_file_name):
        # read the 'model' and 'scaler' files which were saved
        self.raw_data = read_data(path_name=data_file_name)
        self.scaler = None
        self.preprocessed_data = None
        self.data_scaling = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.reg = None
        self.features = None
        self.targets = None

    def load_and_clean_data(self):
        # Step 1: have a quick glance
        df = re_format_file(df=self.raw_data).copy()
        data_dict = generate_data_dict(df=df)
        # add feature price_per_square_foot
        df['price_per_square_foot'] = df.apply(lambda x: x['saleprice'] / x['lotarea'], axis=1)

        # Step 2: handle missing value (chú ý: chọn value_to_fill trước khi chaỵ: mean, mode, other level)
        df = handle_missing_value(df=df)

        # Step 4: handle outliner (chú ý: drop outliner by z-score or drop outliner by iqr in case type = int)

        # column_int_type = df.select_dtypes(["number"]).columns
        # if dependent_variable in column_int_type:
        #     df = drop_outliner_by_zscore(df=df, dependent_variable=dependent_variable)
        #     # df = drop_outliner_by_iqr(df=df)
        # else:
        #     pass
        #
        # # Step 5: removing highly correlated variable
        # df = df.drop(columns=['z_column_name'])
        # # df = df.drop(columns=[dependent_variable])
        # # df = simple_correlated_detection(df=df)
        # # Step 6: one hot coding: drop_first = True để bỏ đi 1 biến khi thực hiện one hot coding
        #
        # df = pd.get_dummies(df, drop_first=True)
        # # # Step 6: reformat_file
        # df = re_format_file(df=df)
        # print('Original data shape:', original_df.shape, '\nFinal data shape:', df.shape)
        # return df

        # self.targets = df['absenteeism_time_in_hours'].apply(
        #     lambda x: 0 if x <= breaking_point else 1)
        # self.features = df.drop(columns_to_remove, axis=1)
        # self.preprocessed_data = self.features.join(self.targets)
        #
        # # Step 5: data scaling
        # scale_columns = [x for x in self.features.columns.values if x not in columns_to_remove]
        # absenteeism_scaler = CustomScaler(columns=scale_columns, copy=True, with_mean=True, with_std=True)
        # absenteeism_scaler.fit(X=self.features, y=None)
        # self.data_scaling = absenteeism_scaler.transform(self.features)

    # step 6: Train/test split
    #     def train_test_split(self):
    #         x_train, x_test, y_train, y_test = train_test_split(self.data_scaling, self.targets, train_size=0.8, shuffle=True,
    #                                                             random_state=20)
    #         self.x_train = x_train
    #         self.x_test = x_test
    #         self.y_train = y_train
    #         self.y_test = y_test
    #         self.reg = LogisticRegression().fit(X=x_train, y=y_train)

    def summary_table(self):
        # Extract intercept and coefficients
        intercept = self.reg.intercept_
        coef = self.reg.coef_[0]
        feature_name = self.features.columns.values

        summary_table = pd.DataFrame(
            {
                'column_name': np.append(['intercept'], feature_name),
                'coefficient': np.append(intercept, coef)
            }
        )
        summary_table['odds_ratio'] = np.exp(summary_table.coefficient)
        summary_table = summary_table.sort_values('odds_ratio', ascending=False)
        print(summary_table)

    def predicted_probability(self, x: list):
        pred = self.reg.predict_proba(x)[:, 1]
        return pred

    # a function which outputs 0 or 1 based on our model
    def predicted_output_category(self, x: list):
        pred_outputs = self.reg.predict(x)
        return pred_outputs

    def save_model(self):
        with open('absenteeism_model', 'wb') as file:
            pickle.dump(self.reg, file)
        with open('absenteeism_scaler', 'wb') as file:
            pickle.dump(self.data_scaling, file)


def linear_regression_ols_logit(linear_regression: bool, df: object, dependent_variable: str, p_value: int = None):
    features_variable = df.columns.tolist()
    features_variable.remove(dependent_variable)

    # Step 2: observe draf OLS regression
    ols_features = ""
    for feature in features_variable:
        ols_features = ols_features + ' + ' + feature
    ols_features = ols_features[3:]
    if linear_regression:
        results = smf.ols(f"{dependent_variable} ~ {ols_features}", data=df).fit()
    else:
        results = smf.logit(f"{dependent_variable} ~ {ols_features}", data=df).fit()
    print(results.summary())
    if not p_value:
        return results
    else:
        ols_features_pvalue = ""
        for fearture_p_value in results.pvalues.index:
            if results.pvalues[fearture_p_value] < p_value:
                ols_features_pvalue = ols_features_pvalue + ' + ' + fearture_p_value
        # note: remove Intercept
        ols_features_pvalue = ols_features_pvalue[15:]
        if linear_regression:
            result_pvalue = smf.ols(f"{dependent_variable} ~ {ols_features_pvalue}", data=df).fit()
        else:
            result_pvalue = smf.logit(f"{dependent_variable} ~ {ols_features_pvalue}", data=df).fit()
        print(result_pvalue.summary())
        return result_pvalue


def feature_selection_randomforest(linear_regression: bool, df: object, dependent_variable: str,
                                   n_top_highest_feature_importance: int = None):
    features_variable = df.columns.tolist()
    features_variable.remove(dependent_variable)

    X = df[features_variable]
    Y = df[dependent_variable]
    names = features_variable
    if linear_regression:
        rf = RandomForestRegressor()
    else:
        rf = RandomForestClassifier()

    rf.fit(X, Y)
    feature_importance = pd.DataFrame(
        {
            'names': names,
            'feature_importance': rf.feature_importances_
        }
    )

    sorted_fi = feature_importance.sort_values(by="feature_importance", ascending=False)
    if not n_top_highest_feature_importance:
        print(sorted_fi)
        return sorted_fi
    else:
        print(sorted_fi.head(n_top_highest_feature_importance))
        return sorted_fi.head(n_top_highest_feature_importance)


def forward_selected(df: object, dependent_variable: str):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(df.columns)
    remaining.remove(dependent_variable)
    print(len(remaining))
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(dependent_variable,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, df).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score

    formula = "{} ~ {} + 1".format(dependent_variable,
                                   ' + '.join(selected))
    print(formula)
    model = smf.ols(formula, df).fit()
    print(model.summary())
    return model


# def linear_regression_diagnostics():
#     print('joy xinh')


if __name__ == "__main__":
    start_time = time.time()
    pd.set_option("display.max_rows", None, "display.max_columns", 30, 'display.width', 500)
    url = "https://raw.githubusercontent.com/tiwari91/Housing-Prices/master/train.csv"

    house_price_model = data_modeling(data_file_name=url)
    house_price_model.load_and_clean_data()

    # https://raw.githubusercontent.com/tiwari91/Housing-Prices/master/train.csv
    # url = "https://github.com/pnhuy/datasets/raw/master/Churn.xls"

    #
    # df_train, df_test = train_test_split(df, test_size=0.3)

    # linear_regression_ols_logit(df=df_train, dependent_variable='saleprice', linear_regression=True)

    # Linear regression by randomforest
    # Step 1: feature_selection
    # sorted_fi = feature_selection_randomforest(linear_regression=True, df=df_train, n_top_highest_feature_importance=20,
    #                                            dependent_variable='saleprice')
    #
    # features_variable = sorted_fi['names'].values.tolist()
    # features_variable.append('saleprice')
    # k = df_train[features_variable].copy().reset_index(drop=True)
    #
    # olsres = linear_regression_ols_logit(linear_regression=True, df=k, dependent_variable='saleprice')

    # Linear regression by stepwise algorithms
    # forward_selected(df=df_train[features_variable], dependent_variable='saleprice')

    # REGRESSION PREDICT
    # pred_y = olsres.predict(df_test[features_variable])
    # df_test_predict = df_test.copy()
    # df_test_predict['predict_y'] = pred_y
    # df_test_predict = df_test_predict[['saleprice', 'predict_y']].sort_values(by=['saleprice', 'predict_y'],ascending=True, ignore_index = True)
    # df_test_predict['predict_y'] = df_test_predict.apply(
    #         lambda x: int(x['predict_y']), axis=1)
    # # print(df_test_predict)
    # df_test_predict.plot.line()
    # plt.show()

    # REGRESSION DIAGNOSTIC
    # from statsmodels.graphics.regressionplots import plot_leverage_resid2
    # fig, ax = plt.subplots(figsize=(8, 6))
    # fig = plot_leverage_resid2(olsres, ax=ax)
    # plt.show()

    # 1. Heteroscedasticity Tests
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white, acorr_breusch_godfrey
    import statsmodels.stats.api as sms

    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # breuschpagan = het_breuschpagan(resid=olsres.resid, exog_het=olsres.model.exog)
    # # het_breuschpagan: return (lm, lm_pvalue, fvalue, f_pvalue)
    # white = het_white(resid=olsres.resid, exog=olsres.model.exog)
    # print("Heteroscedasticity_breuschpagan: return (lm, lm_pvalue, fvalue, f_pvalue)", breuschpagan)
    # print("Heteroscedasticity_white: return (lm, lm_pvalue, fvalue, f_pvalue)", white)

    # 2. Autocorrelation Tests
    # breusch_godfrey = acorr_breusch_godfrey(res=olsres)
    # print("Autocorrelation_breusch_godfrey: return (lm, lm_pvalue, fvalue, f_pvalue)", breusch_godfrey)
    # 3. normal distribution residual
    # name = ['Jarque-Bera', 'Jarque-Bera_pvalue', 'Skew', 'Kurtosis']
    # jarque_bera = sms.jarque_bera(olsres.resid)
    # omni_normtest = sms.omni_normtest(olsres.resid)
    # print("normal distribution residual jarque_bera: return ('Jarque-Bera', 'Jarque-Bera_pvalue', 'Skew', 'Kurtosis') ", jarque_bera)
    # print("normal distribution residual omni_normtest: ", omni_normtest)

    # 4. Multicollinearity Tests

    # CLASSIFICATION
    # CACH 1: CLASSIFICATION VOI statmodel bang thuat toan MLE(Maximum Likelyhood Estimation)

    # Step 1: feature_selection
    # logreg = linear_regression_ols_logit(linear_regression=False, df=df_train, dependent_variable='churn', p_value=0.05)
    # pred_proba = logreg.predict(df_test)

    # # Set threshold to predict
    # pred_label = (pred_proba > 0.5).astype('int')
    # test_result = pd.concat([df_test, pred_label], axis=1)
    # # print(test_result[['account_length', 'churn', 0]])
    #
    # print('Accuracy:', accuracy_score(df_test['churn'], pred_label))
    # print('Precision:', precision_score(df_test['churn'], pred_label))
    # print('Recall:', recall_score(df_test['churn'], pred_label))
    # print('F1-score:', f1_score(df_test['churn'], pred_label))
    # print('AUC:', roc_auc_score(df_test['churn'], pred_proba))

    # CACH 2: CLASSIFICATION VOI sklearn bang randomforest

    # X_train = df_train.drop(columns='churn')
    # y_train = df_train.churn
    #
    # X_test = df_test.drop(columns='churn')
    # y_test = df_test.churn
    # # Creat gridsearch
    # params = {
    #     'n_estimators': [50, 200, 300, 500, 700],
    #     'max_depth': [None],
    #     'min_samples_split': [2, 5, 10]
    # }
    #
    # rf = RandomForestClassifier()
    # rf.fit(X_train, y_train)
    # pred_proba = rf.predict_proba(X_test)[:, 1]
    # pred_label = rf.predict(X_test)
    # df_test_predict = df_test.reset_index()[['account_length', 'churn']]
    # df_test_predict['predict_proba'] = pred_proba
    # df_test_predict['predict_label'] = pred_label
    # print(df_test_predict)
    # print(test_result[['account_length', 'churn', 0]])
    #
    # print('Accuracy:', accuracy_score(df_test.churn, pred_label))
    # print('Precision:', precision_score(df_test.churn, pred_label))
    # print('Recall:', recall_score(df_test.churn, pred_label))
    # print('F1-score:', f1_score(df_test.churn, pred_label))
    # print('AUC:', roc_auc_score(df_test.churn, pred_label))

    # grid = GridSearchCV(rf, param_grid=params)
    # grid.fit(X=X_train, y=y_train)
    # k = grid.best_params_
    # print(k)
    # best_rf = grid.best_estimator_
    # best_rf.fit(X_train, y_train)
    # best_pred_label = best_rf.predict(X_test)
    #
    # print('Accuracy:', accuracy_score(df_test.churn, best_pred_label))
    # print('Precision:', precision_score(df_test.churn, best_pred_label))
    # print('Recall:', recall_score(df_test.churn, best_pred_label))
    # print('F1-score:', f1_score(df_test.churn, best_pred_label))
    # print('AUC:', roc_auc_score(df_test.churn, best_pred_label))

    # CLASSIFICATION
    # CACH 3: CLASSIFICATION VOI xgboost
    # X_train = df_train.drop(columns='churn')
    # y_train = df_train.churn
    # X_test = df_test.drop(columns='churn')
    # y_test = df_test.churn

    # xg_reg = xgb.XGBClassifier()
    # xg_reg.fit(y=y_train, X=X_train)

    # y_test_pred = xg_reg.predict(X_test)
    # pred_proba = xg_reg.predict_proba(X_test)[:, 1]
    # pred_label = xg_reg.predict(X_test)
    # df_test_predict = df_test.reset_index()[['account_length', 'churn']]
    # df_test_predict['predict_proba'] = pred_proba
    # df_test_predict['predict_label'] = pred_label

    # print(df_test_predict)

    # print('Accuracy xgboost:', accuracy_score(df_test.churn, pred_label))
    # print('Precision xgboost:', precision_score(df_test.churn, pred_label))
    # print('Recall xgboost:', recall_score(df_test.churn, pred_label))
    # print('F1-score xgboost:', f1_score(df_test.churn, pred_label))
    # print('AUC xgboost:', roc_auc_score(df_test.churn, pred_label))

    print("\n --- total time to process %s seconds ---" % (time.time() - start_time))
