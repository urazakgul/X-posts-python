import yfinance as yf
import pandas as pd
import numpy as np
from requests_html import HTMLSession
from dtaidistance import dtw
import matplotlib.pyplot as plt

def fetch_crypto_data(num_currencies=10):
    session = HTMLSession()
    resp = session.get(f"https://finance.yahoo.com/crypto?offset=0&count={num_currencies}")
    tables = pd.read_html(resp.html.raw_html)
    df = tables[0].copy()
    tickers = df.Symbol.tolist()

    dfs = []
    for ticker in tickers:
        data = yf.download(tickers=ticker, start='2000-01-01', interval='1d', progress=False)
        data = data[['Close']].rename(columns={'Close': ticker})
        dfs.append(data)

    merged_df = pd.concat(dfs, axis=1)

    return merged_df

def calculate_dtw_distances(merged_df, target_currency):
    non_nan_counts = merged_df.count()
    target_non_nan_count = non_nan_counts[target_currency]

    merged_df = merged_df.loc[:, merged_df.count() >= target_non_nan_count]

    dtw_scores = pd.DataFrame(
        index=merged_df.columns,
        columns=merged_df.columns,
        dtype=np.float64
    )

    for col1 in merged_df.columns:
        for col2 in merged_df.columns:

            col1_values = merged_df[col1].dropna().values[:target_non_nan_count]
            col2_values = merged_df[col2].dropna().values[:target_non_nan_count]

            col1_values_normalized = np.log(col1_values)
            col2_values_normalized = np.log(col2_values)

            distance = dtw.distance(col1_values_normalized, col2_values_normalized)
            dtw_scores.loc[col1, col2] = distance

    dtw_scores_stacked = dtw_scores.stack().sort_values()
    dtw_df = dtw_scores_stacked.reset_index()
    dtw_df.columns = ['Crypto1', 'Crypto2', 'DTW Distance']
    dtw_df_unique = dtw_df[dtw_df['Crypto1'] != dtw_df['Crypto2']]
    dtw_df_unique = dtw_df_unique.drop_duplicates(subset='DTW Distance')
    dtw_df_unique = dtw_df_unique.reset_index(drop=True)

    target_index = dtw_df_unique[(dtw_df_unique['Crypto1'] == target_currency) | (dtw_df_unique['Crypto2'] == target_currency)].index[0]
    selected_crypto1 = dtw_df_unique.iloc[target_index]['Crypto1']
    selected_crypto2 = dtw_df_unique.iloc[target_index]['Crypto2']

    return dtw_df_unique, selected_crypto1, selected_crypto2

def plot_crypto_prices(merged_df, selected_crypto1, selected_crypto2):
    data1 = np.log(merged_df[selected_crypto1].dropna().reset_index(drop=True))
    data2 = np.log(merged_df[selected_crypto2].dropna().reset_index(drop=True))

    plt.figure(figsize=(10, 6))
    plt.plot(data1, label=selected_crypto1)
    plt.plot(data2, label=selected_crypto2)
    plt.title(f'Closing Prices of {selected_crypto1} and {selected_crypto2}')
    plt.xlabel('Days')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    target_currency = 'SOL-USD'

    merged_df = fetch_crypto_data(num_currencies=10)
    dtw_df_unique, selected_crypto1, selected_crypto2 = calculate_dtw_distances(merged_df, target_currency)
    plot_crypto_prices(merged_df, selected_crypto1, selected_crypto2)