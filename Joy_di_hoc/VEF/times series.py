import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA, ARIMA
from Joy_di_hoc.VEF.practice_day1 import prepare_data

if __name__ == "__main__":
    start_time = time.time()
    pd.set_option("display.max_rows", None, "display.max_columns", 30, 'display.width', 500)
    AIR = "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
    original_df = pd.read_csv(AIR)
    original_df['Month'] = pd.to_datetime(original_df['Month'], format='%Y-%m')
    original_df = original_df.set_index('Month')
    '''
        Dickey_Fuller_Test:
            Ho: Non-stationary
            H1: Stationary
        Return:
            adf: float        
            pvalue: float            
            usedlag: int    
            nobs: int
            critical values: dict            
            icbest: float        
            resstore: ResultStore, optional
        Source: https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html        
    '''
    # Dickey_Fuller_Test = adfuller(original_df)
    # print(Dickey_Fuller_Test)
    # plt.figure(figsize=(15, 9))
    # original_df.plot()
    # plt.show()

    # First we penalize higher values by taking log
    log_passengers = np.log(original_df)
    # log_passengers.plot(figsize=(15, 6))
    # plt.show()

    # Take 1st differencing on transformed time series
    log_passengers_diff_1 = log_passengers - log_passengers.shift(1)
    log_passengers_diff_1.dropna(inplace=True)  # drop NA values
    log_passengers_diff_1.plot(figsize=(15, 6))
    plt.show()



    print("\n --- total time to process %s seconds ---" % (time.time() - start_time))