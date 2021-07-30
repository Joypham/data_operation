import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans
from Joy_di_hoc.VEF.practice_day1 import prepare_data

if __name__ == "__main__":
    start_time = time.time()
    pd.set_option("display.max_rows", None, "display.max_columns", 30, 'display.width', 500)
    DEMO_DATA = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Howell1.csv"
    original_df = pd.read_csv(DEMO_DATA, sep=";")
    df = prepare_data(original_df=original_df, dependent_variable='weight')
    demo = prepare_data(original_df=df, dependent_variable='height')

    df = demo[(demo.male == 1) & (demo.age <= 50) & (demo.age >= 18)].reset_index()
    # print(df)
    # plt.scatter(x='weight', y='height', data=df)
    # plt.show() # có vẻ tuyến tính đây :)))

    kmeans = KMeans(n_clusters=4)
    X = df[['weight', 'height']]
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    df['predict_cluster'] = y_kmeans

    # plt.scatter(x='weight', y='height', data=df, c = y_kmeans)
    # plt.show()

    # calculate optimize n_cluster with Ebowl chart
    # distortions = []
    # ks = range(1, 10)
    # for k in ks:
    #     kmeans = KMeans(n_clusters=k)
    #     kmeans.fit(X)
    #     distortions.append(kmeans.inertia_)
    # plt.plot(ks, distortions, '-x')
    # plt.show()  # nguyên lý khuyủ tay: số lượng cluster hợp lý là số ở điểm khuỷu tay: trong trường họp này k = 3



    # print(df.head(10))

    print("\n --- total time to process %s seconds ---" % (time.time() - start_time))