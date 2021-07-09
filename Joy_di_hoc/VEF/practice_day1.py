import numpy as np
import pandas as pd
from scipy.stats import mode, zscore
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import re
import time
from google_spreadsheet_api.function import get_df_from_speadsheet, get_list_of_sheet_title, update_value, \
    creat_new_sheet_and_update_data_from_df, get_gsheet_name
DATA = "https://raw.githubusercontent.com/tiwari91/Housing-Prices/master/train.csv"


def original_df():
    df = pd.read_csv(DATA)
    # reformat_column namr: lowercase entire column name (statmodel ko doc duoc column name trong 1 so truong hop)

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
    url = "https://docs.google.com/spreadsheets/d/1cZw8dBSCJF1ylVakiqC5oKHIaEvuPV6zid2ct07Bo4k"
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
    # df[numerical_data_column] = df[numerical_data_column].fillna(df[numerical_data_column].mean())

    # fill NA numerical_column by median:
    df[numerical_data_column] = df[numerical_data_column].fillna(df[numerical_data_column].median())

    # fill NA non_numerical_column by mode:
    for column_name in non_numerical_data_column:
        df[column_name] = df[column_name].fillna(mode(df[column_name])[0][0])

    # fill NA non_numerical_column by other level:
    # df[non_numerical_data_column] = df[non_numerical_data_column].fillna("unknown")
    return df


def drop_outliner_by_zscore(df: object, column_name: str = None):
    '''
       z-socre: tính khoảng cách từ 1 điểm đến điểm trung bình có hiệu chỉnh theo std
    '''

    z_column_name = zscore(df['saleprice'])
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


def prepare_data():
    # Step 1: reformat and read file
    df = original_df()

    # Step 2: observe data dict
    # generate_data_dict(df=df)

    # Step 3: handle missing value (chú ý: chọn value_to_fill trước khi chaỵ: mean, mode, other level)
    df = handle_missing_value(df=df)

    # Step 4: handle outliner (chú ý: drop outliner by z-score or drop outliner by iqr)
    df = drop_outliner_by_zscore(df=df)
    # df = drop_outliner_by_iqr(df=df)

    # Step 5: removing highly correlated variable
    df = df.drop(columns=['z_column_name', 'id'])
    df = simple_correlated_detection(df=df)

    # Step 5: one hot coding: drop_first = True để bỏ đi 1 biến khi thực hiện one hot coding
    df = pd.get_dummies(df, drop_first=True)

    # Step 6: reformat_file
    df = re_format_file(df=df)
    print('Original data shape:', original_df().shape, '\nFinal data shape:', df.shape)

    return df


def linear_regression_ols(df: object):
    # Step 1:
    numerical_data_column = simple_correlated_detection(df=original_df().drop(columns=['id', 'saleprice'])).select_dtypes("number").columns
    new_numerical_data_column = []
    for i in numerical_data_column:
        # ky tự đặc biệt chuyển hết về _
        # start_with = con số: thêm _ đằng trước nó
        new_name = re.sub(r'\W+', '_', i)

        if re.match('^\d', new_name):
            new_name = '_' + new_name
        new_numerical_data_column.append(new_name)

    # Step 2: observe draf OLS regression
    ols_features = ""
    for feature in new_numerical_data_column:
        ols_features = ols_features + ' + ' + feature
    ols_features = ols_features[3:]
    results = smf.ols(f"saleprice ~ {ols_features}", data=df).fit()
    # print(results.summary())

    # Step 3: fearture selection

    X = df[new_numerical_data_column]
    Y = df['saleprice']
    names = new_numerical_data_column
    rf = RandomForestRegressor()
    rf.fit(X, Y)
    feature_importance = pd.DataFrame(
        {
            'names': names,
            'feature_importance': rf.feature_importances_
        }
    )

    sorted_fi = feature_importance.sort_values(by="feature_importance", ascending=False)
    print(sorted_fi)
    # top_feature = sorted_fi['names'].head(15).values.tolist()
    # print(top_feature)


if __name__ == "__main__":
    start_time = time.time()
    pd.set_option("display.max_rows", None, "display.max_columns", 30, 'display.width', 500)
    df = prepare_data()
    df1 = df.copy()
    linear_regression_ols(df=df1)

    print("\n --- total time to process %s seconds ---" % (time.time() - start_time))
