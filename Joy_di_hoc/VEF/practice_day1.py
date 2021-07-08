import numpy as np
import pandas as pd
from scipy.stats import mode, zscore
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
    house_prices = pd.read_csv(DATA)
    return house_prices


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


def prepare_data(df: object):
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

    # fill NA non_numerical_column by mode:
    for column_name in non_numerical_data_column:
        df[column_name] = df[column_name].fillna(mode(df[column_name])[0][0])
    print(df.head(100))


if __name__ == "__main__":
    start_time = time.time()
    pd.set_option("display.max_rows", None, "display.max_columns", 30, 'display.width', 500)
    df = original_df()
    # generate_data_dict(df=df)
    prepare_data(df=df)

    print("\n --- total time to process %s seconds ---" % (time.time() - start_time))
