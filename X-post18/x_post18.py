import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import mplfinance as mpf

def kmeans_resistance_support(ticker, start_date, n_clusters):

    df = yf.download(
        tickers=ticker,
        start=start_date,
        progress=False
    )[['Open','High','Low','Adj Close']]

    df = df.rename(columns={'Adj Close':'Close'})

    log_close_prices = np.log(df['Close'])

    time_feature = np.linspace(0, 1, len(df)).reshape(-1, 1)
    clustering_features = np.column_stack((time_feature, log_close_prices))

    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(clustering_features)

    exp_cluster_centers = np.exp(kmeans.cluster_centers_[:, 1])

    last_price = df['Close'].iloc[-1]

    support_levels = exp_cluster_centers[exp_cluster_centers < last_price]
    resistance_levels = exp_cluster_centers[exp_cluster_centers > last_price]

    if len(support_levels) > 0:
        support_level = "{:.2f}".format(support_levels[-1])
    else:
        support_level = None

    if len(resistance_levels) > 0:
        resistance_level = "{:.2f}".format(resistance_levels[0])
    else:
        resistance_level = None

    print(f"Support Level: {support_level}, Resistance Level: {resistance_level}")

    mpf.plot(
        data=df,
        type='candle',
        style='yahoo',
        hlines=dict(
            hlines=exp_cluster_centers.tolist(),
            linestyle='-.',
            colors='red'
        ),
        figscale=1.5,
        title=f"Identifying Support and Resistance Levels using K Means Clustering for {ticker}"
    )

ticker='BTC-USD'
start_date='2023-09-01'

kmeans_resistance_support(ticker, start_date, n_clusters=10)