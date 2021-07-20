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
from google_spreadsheet_api.function import get_df_from_speadsheet, get_list_of_sheet_title, update_value, \
    creat_new_sheet_and_update_data_from_df, get_gsheet_name


def original_df(url: str):

    if ".xls" in url:
        df = pd.read_excel(url)
    else:
        df = pd.read_csv(url)
    # reformat_column name: lowercase entire column name (statmodel ko doc duoc column name trong 1 so truong hop)

    lower_names = [name.lower() for name in df.columns]
    df.columns = lower_names
    return df


def generate_data_dict(df: object):
    mean = df.mean(axis=0)
    median = df.median()
    std = df.std()
    result1 = {
        "mean": mean,
        "median": median,
        "std": std,
    }
    data_dict_1 = pd.DataFrame(result1).reset_index()

    count_distinct = df.nunique()
    n_missing = pd.isnull(df).sum()
    n_zeros = (df == 0).sum()
    mode_value = mode(df)[0][0]
    mode_count = mode(df)[1][0]
    result2 = {
        "count_distinct": count_distinct,
        "n_missing": n_missing,
        "n_zeros": n_zeros,
        "mode_value": mode_value,
        "mode_count": mode_count,
    }
    data_dict_2 = pd.DataFrame(result2).reset_index()

    data_dict_merge = pd.merge(data_dict_2, data_dict_1, how='left', on='index',
                               validate='1:m').fillna(value='None')
    data_dict_merge.columns = data_dict_merge.columns.str.replace('index', 'column_name')

    # Write in gsheet
    # creat_new_sheet_and_update_data_from_df(df=data_dict_merge,
    #                                         gsheet_id="1cZw8dBSCJF1ylVakiqC5oKHIaEvuPV6zid2ct07Bo4k",
    #                                         new_sheet_name="data dict")
    # url = "https://docs.google.com/spreadsheets/d/1cZw8dBSCJF1ylVakiqC5oKHIaEvuPV6zid2ct07Bo4k"
    # print(url, "\n\n", data_dict_merge)
    return data_dict_merge


def handle_missing_value(df: object):
    # Get column_types
    numerical_data_column = df.select_dtypes("number").columns
    non_numerical_data_column = df.select_dtypes(["object"]).columns


    '''
        missing value:
            - numerical_column: mean, median, build predict model
            - non_numerical_column: mode, other level ,build predict model 
    '''
    # fill NA numerical_column by mean
    df[numerical_data_column] = df[numerical_data_column].fillna(df[numerical_data_column].mean())

    # fill NA numerical_column by median:
    # df[numerical_data_column] = df[numerical_data_column].fillna(df[numerical_data_column].median())

    # fill NA non_numerical_column by mode:
    # for column_name in non_numerical_data_column:
    #     df[column_name] = df[column_name].fillna(mode(df[column_name])[0][0])

    # fill NA non_numerical_column by other level:
    df[non_numerical_data_column] = df[non_numerical_data_column].fillna("unknown")
    return df


def drop_outliner_by_zscore(df: object, dependent_variable: str):
    '''
       z-socre: tính khoảng cách từ 1 điểm đến điểm trung bình có hiệu chỉnh theo std
    '''
    z_column_name = zscore(df[dependent_variable])

    df['z_column_name'] = z_column_name
    df_outliner = df[((df.z_column_name > 2) | (df.z_column_name < -2))]
    print('The number of outliers:', len(df_outliner.index))
    index_outliner = df_outliner.index
    df = df.drop(index_outliner)
    return df


def drop_outliner_by_iqr(df: object, column_name: str = None):
    '''
        - IQR là 1 trong những cách để detect outliner nhưng là cách tệ nhất :)))
    '''

    q75, q25 = np.percentile(df['saleprice'], [75, 25])
    iqr = q75 - q25
    upper_whisker = q75 + 1.5 * iqr
    lower_whisker = q25 - 1.5 * iqr
    df_outliner = df[((df.saleprice >= upper_whisker) | (df.saleprice <= lower_whisker))]
    index_outliner = df_outliner.index
    df = df.drop(index_outliner)
    return df


def re_format_file(df: object):
    new_columns = []
    for i in df.columns:
        # ky tự đặc biệt chuyển hết về _
        # start_with = con số: thêm _ đằng trước nó
        new_name = re.sub(r'\W+', '_', i)

        if re.match('^\d', new_name):
            new_name = '_' + new_name
        new_columns.append(new_name)
    df.columns = new_columns
    return df


def simple_correlated_detection(df):
    # bỏ biến phụ thuộc (saleprice: vì mình đang dự đoán biến này)
    corr_matrix = df.corr().abs()
    # bỏ đi phần đường chéo đối xứng
    corr_matrix_reformat = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # lấy ra column có corr > 0.8
    to_drop = [column for column in corr_matrix_reformat.columns if any(corr_matrix_reformat[column] > 0.8)]
    '''
        - Các cặp biến có corr > 0.8:
            + [garagearea,garagecars]
            + [totrmsabvgrd,grlivarea]
        - Lựa chọn giữ lại 1 biến đưa vào mô hình
    '''
    print('Highly correlated variables to drop:', to_drop)
    df = df.drop(columns=to_drop)
    return df


def prepare_data(url: str, dependent_variable: str):
    # Step 1: reformat and read file
    df = original_df(url=url)

    # Step 2: observe data dict
    # generate_data_dict(df=df)

    # Step 3: handle missing value (chú ý: chọn value_to_fill trước khi chaỵ: mean, mode, other level)
    df = handle_missing_value(df=df)

    # Step 4: handle outliner (chú ý: drop outliner by z-score or drop outliner by iqr in case type = int)
    column_int_type = df.select_dtypes(["number"]).columns
    if dependent_variable in column_int_type:
        drop_outliner_by_zscore(df=df, dependent_variable=dependent_variable)
        # df = drop_outliner_by_iqr(df=df)
    else:
        pass

    # Step 5: removing highly correlated variable
    df = df.drop(columns=['z_column_name', 'id'])
    # df = df.drop(columns=[dependent_variable])
    df = simple_correlated_detection(df=df)
    # # Step 6: one hot coding: drop_first = True để bỏ đi 1 biến khi thực hiện one hot coding
    df = pd.get_dummies(df, drop_first=True)
    # # Step 6: reformat_file
    df = re_format_file(df=df)
    print('Original data shape:', original_df(url=url).shape, '\nFinal data shape:', df.shape)
    return df


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


if __name__ == "__main__":
    start_time = time.time()
    pd.set_option("display.max_rows", None, "display.max_columns", 30, 'display.width', 500)
    url = "https://raw.githubusercontent.com/tiwari91/Housing-Prices/master/train.csv"
    # url = "https://github.com/pnhuy/datasets/raw/master/Churn.xls"
    df = prepare_data(url=url, dependent_variable='saleprice')
    df_train, df_test = train_test_split(df, test_size=0.3)

    # linear_regression_ols_logit(df=df_train, dependent_variable='saleprice', linear_regression=True)

    # Linear regression by randomforest
    # Step 1: feature_selection
    sorted_fi = feature_selection_randomforest(linear_regression=True, df=df_train, n_top_highest_feature_importance=20,
                                               dependent_variable='saleprice')

    features_variable = sorted_fi['names'].values.tolist()
    features_variable.append('saleprice')
    print(features_variable)
    linear_regression_ols_logit(linear_regression=True, df=df_train[features_variable], dependent_variable='saleprice')

    # Linear regression by stepwise algorithms
    # forward_selected(df=df, dependent_variable='saleprice')

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
