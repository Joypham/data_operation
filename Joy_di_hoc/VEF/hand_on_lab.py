# https://docs.google.com/spreadsheets/d/1iEZRdDJJ3bu0BIrjgW7MTlVWsbszpIo-Lr0Ca5Da6N4/edit#gid=854983440
from google_spreadsheet_api.function import get_df_from_speadsheet, get_list_of_sheet_title, update_value, \
    creat_new_sheet_and_update_data_from_df, get_gsheet_name
import time
import pandas as pd
import numpy as np
from scipy.stats import mode, zscore
import matplotlib.pyplot as plt
import seaborn
import re

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def get_gsheet_id_from_url(url: str):
    url_list = url.split("/")
    gsheet_id = url_list[5]
    return gsheet_id


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
    creat_new_sheet_and_update_data_from_df(df=data_dict_merge,
                                            gsheet_id="1cZw8dBSCJF1ylVakiqC5oKHIaEvuPV6zid2ct07Bo4k",
                                            new_sheet_name="data dict")
    # print(data_dict_merge)
    return data_dict_merge


def missing_value(df: object):
    # Numerical and non numerical column_name

    numerical_data_column = df.select_dtypes("number").columns
    non_numerical_data_column = df.select_dtypes(["object"]).columns

    # Missing value
    # Numerical column_name: fullfilled by mean()
    df[numerical_data_column] = df[numerical_data_column].fillna(df[numerical_data_column].mean())

    # Non numerical column_name: fullfilled by mode()
    # Non numerical column_name: fullfilled by other level
    for column_name in non_numerical_data_column:
        df[column_name] = df[column_name].fillna(mode(df[column_name])[0][0])
    return df


def drop_outliner_by_zscore(df: object, column_name: str):
    z_column_name = zscore(df['saleprice'])
    df['z_column_name'] = z_column_name
    df_outliner = df[((df.z_column_name > 2) | (df.z_column_name < -2))]
    index_outliner = df_outliner.index
    df = df.drop(index_outliner)
    return df


def prepare_data(df: object):
    numerical_data_column = df.select_dtypes("number").columns
    non_numerical_data_column = df.select_dtypes(["object"]).columns

    df_n_missing = missing_value(df=df)
    drop_outliner_by_zscore(df=df_n_missing, column_name='saleprice')


if __name__ == "__main__":
    start_time = time.time()
    pd.set_option("display.max_rows", None, "display.max_columns", 50, 'display.width', 1000)

    train_path = "https://raw.githubusercontent.com/tiwari91/Housing-Prices/master/train.csv"

    df = pd.read_csv(train_path)
    # lowercase entire column name
    lower_names = [name.lower() for name in df.columns]
    # match column_name = lower column_name
    df.columns = lower_names
    # change dtype to save memory usage and performance: memory usage from 924 => 874 KB
    df = df.astype({'overallqual': 'int8',
                    'overallcond': 'int8',
                    'fireplaces': 'int8',
                    'garagecars': 'int8',
                    'mosold': 'int8'
                    })

    # k = generate_data_dict(df=df)
    prepare_data(df=df)

    # drop outliner have abs(z-score) > 2
    # df = drop_outliner_by_zscore(df=df, column_name='saleprice')

    # remove variables have highly corr
    # bỏ biến phụ thuộc (saleprice: vì mình đang dự đoán biến này)
    # df = df.drop(columns=['z_column_name'])
    # corr_matrix = df.corr().abs()
    # # bỏ đi phần đường chéo đối xứng
    # upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # # lấy ra column có corr > 0.8
    # to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
    # df = df.drop(columns=to_drop)

    # One hot coding
    # Drop_first = True để bỏ đi 1 biến khi thực hiện one hot coding
    # df = pd.get_dummies(df, drop_first=True)
    # print(df)
    # new_columns = []
    # for i in df.columns:
    #     # ky tự đặc biệt chuyển hết về _
    #     # start_with = con số: thêm _ đằng trước nó
    #     new_name = re.sub(r'\W+', '_', i)
    #
    #     if re.match('^\d', new_name):
    #         new_name = '_' + new_name
    #     new_columns.append(new_name)
    # df.columns = new_columns

    # Linear regression: su dung phuong phap OLS
    # results = smf.ols('saleprice ~ overallqual + lotarea', data=df).fit()
    # print(results.summary())

    # RandomForest
    # Lấy ra các feature quan trọng, bỏ biến id vì ko có ý nghĩa trong phân tích
    # X = df.drop(columns=['saleprice', 'id'])
    # Y = df.saleprice
    # names = X.columns
    # rf = RandomForestRegressor()
    # rf.fit(X, Y)
    #
    # feature_importance = pd.DataFrame(
    #     {
    #         'names': names,
    #         'feature_importance': rf.feature_importances_
    #     }
    # )

    # sorted_fi = feature_importance.sort_values(by="feature_importance", ascending=False)
    # top_feature = sorted_fi['names'].head(15).values.tolist()
    # print(top_feature)




    # Built model dựa vào top_feature

    # df = df[top_feature + ['saleprice']]

    # Viết dài quá, sử dụng hàm join để rút ngắn:
        # ols_features = ""
        # for feature in top_feature:
        #     ols_features = ols_features + ' + ' + feature
        # ols_features = ols_features[3:]
        # print(ols_features+"\n")

    # ols_features = ' + '.join(top_feature)
    # results = smf.ols(f"saleprice ~ {ols_features}", data=df).fit()
    # print(results.summary())


    #


    # Lấy ra n feature có feature_importance cao nhất



    # Histogram
    # plt.hist(df['saleprice'], bins=100)
    # plt.show()

    # Exploratory analysis
    # scatter plot overallQual vs SalePrice: dự đoán mối quan hệ của từng biến với biến mục tiêu
    # plt.scatter(x=df.overallqual, y=df.saleprice)

    # boxplot with seaborn
    # seaborn.catplot(data=df, x="overallqual", y="saleprice", kind="box")

    # Correlation coefficient
    # corr_matrix = df.corr()
    # # heatmap
    # corr_heat = seaborn.heatmap(corr_matrix)
    # plt.show()

    # outliner_row_index = []
    # for row_index in range(len(z_saleprirce)):
    #     if abs(z_saleprirce[row_index]) > 2:
    #         outliner_row_index.append(row_index)
    #         print(row_index, z_saleprirce[row_index])

    # outliner by z-score:
    # print(outliner_row_index)
    # outliner_df = df.loc[outliner_row_index]
    # print(outliner_df)

    # plt.hist(z_saleprirce, bins=30)
    # plt.show()

    # get point have z-score > 2

    print("--- %s seconds ---" % (time.time() - start_time))
