import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.graphics.gofplots import qqplot


def Box_plots(df):
    plt.figure(figsize=(10, 4))
    plt.title("Box Plot")
    sns.boxplot(df)
    plt.show()


def hist_plots(df):
    # plt.figure(figsize=(20, 4))
    # plt.hist(df)
    # plt.title("Histogram Plot")
    # plt.show()

    # plt.figure(figsize=(80, 40))
    _ = sns.histplot(df, x='saleprice', bins=100)
    plt.show()




def scatter_plots(df1, df2):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(df1, df2)
    ax.set_xlabel('Age')
    ax.set_ylabel('Fare')
    plt.title("Scatter Plot")
    plt.show()


def dist_plots(df):
    plt.figure(figsize=(10, 4))
    sns.distplot(df)
    plt.title("Distribution plot")
    sns.despine()
    plt.show()


def qq_plots(df):
    plt.figure(figsize=(10, 4))
    qqplot(df, line='s')
    plt.title("Normal QQPlot")
    plt.show()


def original_df(url: str):
    if ".xls" in url:
        df = pd.read_excel(url)
    else:
        df = pd.read_csv(url, sep=",")
    # reformat_column name: lowercase entire column name (statmodel ko doc duoc column name trong 1 so truong hop)

    lower_names = [name.lower() for name in df.columns]
    df.columns = lower_names
    return df


if __name__ == "__main__":
    import time

    start_time = time.time()
    pd.set_option("display.max_rows", None, "display.max_columns", 30, 'display.width', 500)
    url = "https://raw.githubusercontent.com/tiwari91/Housing-Prices/master/train.csv"
    df = original_df(url=url)
    joy = df.groupby(by=['neighborhood', 'lotfrontage']).count()
    print(joy)
    joy = joy.sort_values(by=['neighborhood', 'lotfrontage'])
    # print(joy)


    # print(df[['lotfrontage','neighborhood']].head(50))
    # hist_plots(df=df)
    # plt.figure(figsize=(8, 12))
    # sns.boxplot(data=df, x="lotfrontage", y="neighborhood")
    # plt.show()


