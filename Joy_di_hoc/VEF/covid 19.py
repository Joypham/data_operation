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
        df = pd.read_csv(url, sep=",")
    # reformat_column name: lowercase entire column name (statmodel ko doc duoc column name trong 1 so truong hop)

    lower_names = [name.lower() for name in df.columns]
    df.columns = lower_names
    return df


def generate_data_dict(df: object):
    mean = df.mean(axis=0)
    min_value = df.min(axis=0, numeric_only=True)
    max_value = df.max(axis=0, numeric_only=True)
    median = df.median()
    std = df.std()
    result1 = {
        "mean": mean,
        "median": median,
        "std": std,
        "min_value": min_value,
        "max_value": max_value

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
                                            new_sheet_name="data_dict_covid_vietnam")
    url = "https://docs.google.com/spreadsheets/d/1cZw8dBSCJF1ylVakiqC5oKHIaEvuPV6zid2ct07Bo4k"
    print(url, "\n\n", data_dict_merge)
    return data_dict_merge


def get_vietnam_covid_data(df: object):
    creat_new_sheet_and_update_data_from_df(df=df,
                                            gsheet_id="1GHwdUYu7qfaNEoustEZ5SqcHaVMjICO7-LE8yoKhTQg",
                                            new_sheet_name="covid_vietnam")


if __name__ == "__main__":
    start_time = time.time()
    pd.set_option("display.max_rows", None, "display.max_columns", 30, 'display.width', 500)
    # source: https://github.com/owid/covid-19-data/tree/master/public/data
    url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    original_df = original_df(url=url)
    vietnam_covid = original_df[original_df['iso_code'] == 'VNM'].reset_index(drop=True).copy()
    # joy = vietnam_covid.fillna('None')
    # creat_new_sheet_and_update_data_from_df(df=joy,
    #                                         gsheet_id="1cZw8dBSCJF1ylVakiqC5oKHIaEvuPV6zid2ct07Bo4k",
    #                                         new_sheet_name="vietnam_covid_today")
    # print(vietnam_covid.fillna)
    # get_vietnam_covid_data(df=vietnam_covid)
    print("\n --- total time to process %s seconds ---" % (time.time() - start_time))
