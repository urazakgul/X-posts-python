import yfinance as yf
import pandas as pd
from dtaidistance import dtw
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

def historical_similar_movements_dtw(df, normalize=True, target_column='Close', target_window_size=30, future_periods=10):

    df = df[[target_column]]

    target_df = df.tail(target_window_size)
    if normalize:
        target_df = (target_df - target_df.min()) / (target_df.max() - target_df.min())

    rolling_df = df.drop(df.tail(target_window_size).index)

    dtw_scores = []

    for i in range(len(rolling_df) - target_window_size + 1):

        sub_rolling_df = rolling_df.iloc[i:(i+target_window_size)]
        if normalize:
            sub_rolling_df = (sub_rolling_df - sub_rolling_df.min()) / (sub_rolling_df.max() - sub_rolling_df.min())

        start_date = sub_rolling_df.index[0]
        end_date = sub_rolling_df.index[-1]

        distance = dtw.distance(
            target_df[target_column].values,
            sub_rolling_df[target_column].values
        )

        dtw_scores.append({
            'Start Date': start_date,
            'End Date': end_date,
            'DTW Score': distance
        })

    dtw_scores = pd.DataFrame(dtw_scores)

    min_score_row = dtw_scores.loc[dtw_scores['DTW Score'].idxmin()]
    start_date = min_score_row['Start Date']
    end_date = min_score_row['End Date']

    similar_df = df.loc[start_date:end_date]
    if normalize:
        similar_df = (similar_df - similar_df.min()) / (similar_df.max() - similar_df.min())

    similar_df_start_index = df.index.get_loc(similar_df.index[0])
    similar_df_end_index = df.index.get_loc(similar_df.index[-1])
    next_n_rows = df.iloc[similar_df_start_index: similar_df_end_index + future_periods + 1]

    target_df = df.tail(target_window_size).reset_index(drop=True)
    next_n_rows = next_n_rows.reset_index(drop=True)

    main_df = pd.merge(next_n_rows, target_df, left_index=True, right_index=True, how='outer')

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(main_df.index, main_df[f'{target_column}_x'], color='tab:blue', label='Similar')
    ax1.set_xlabel('')
    ax1.set_ylabel('Similar', color='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(main_df.index, main_df[f'{target_column}_y'], color='tab:red', label='Target')
    ax2.set_ylabel('Target', color='tab:red')

    plt.title(
        f"Possible {future_periods}-Period Movement of ${ticker.split("-")[0]} Based on Similar Past Movements [Interval: {interval}]",
        fontsize=16
    )

    plt.text(
        0.95,
        0.05,
        'For educational purposes only',
        fontsize=10,
        fontstyle ='italic',
        color='gray',
        ha='right',
        va='bottom',
        transform=ax1.transAxes
    )

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.grid(True)
    plt.show()

    return dtw_scores

ticker='BTC-USD'
start_date='2014-01-01'
interval='1d'

df = yf.download(
    tickers=ticker,
    start=start_date,
    interval=interval,
    progress=False
)

dtw_scores = historical_similar_movements_dtw(df)