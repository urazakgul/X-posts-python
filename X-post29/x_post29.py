import yfinance as yf
import pandas as pd
import numpy as np
from dtaidistance import dtw
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

def historical_similar_movements_dtw(df, normalize=True, distance='dtw', target_column='Close', target_window_size=[20,30,40,50], future_periods=10):

    # The distance parameter must be either 'dtw' or 'euclidean'

    df = df[[target_column]]

    dtw_scores = []
    euclidean_scores = []

    for tws in target_window_size:

        target_df = df.tail(tws)
        if normalize:
            target_df = (target_df - target_df.min()) / (target_df.max() - target_df.min())

        rolling_df = df.drop(df.tail(tws).index)

        for i in range(len(rolling_df) - tws + 1):

            sub_rolling_df = rolling_df.iloc[i:(i+tws)]
            if normalize:
                sub_rolling_df = (sub_rolling_df - sub_rolling_df.min()) / (sub_rolling_df.max() - sub_rolling_df.min())

            start_date = sub_rolling_df.index[0]
            end_date = sub_rolling_df.index[-1]

            if distance == 'dtw':

                dtw_distance = dtw.distance(
                    target_df[target_column].values,
                    sub_rolling_df[target_column].values
                )

                dtw_scores.append({
                    'Start Date': start_date,
                    'End Date': end_date,
                    'Score': dtw_distance,
                    'TWS': tws
                })

            elif distance == 'euclidean':

                euclidean_distance = np.linalg.norm(
                    target_df[target_column].values - sub_rolling_df[target_column].values
                )

                euclidean_scores.append({
                    'Start Date': start_date,
                    'End Date': end_date,
                    'Score': euclidean_distance,
                    'TWS': tws
                })

    distance_scores = pd.DataFrame(dtw_scores) if distance == 'dtw' else pd.DataFrame(euclidean_scores)

    min_score_row = distance_scores.sort_values('Score').iloc[0]
    best_start_date = min_score_row['Start Date']
    best_end_date = min_score_row['End Date']
    best_tws = min_score_row['TWS']

    similar_df = df.loc[best_start_date:best_end_date]
    similar_df_start_index = df.index.get_loc(similar_df.index[0])
    similar_df_end_index = df.index.get_loc(similar_df.index[-1])

    target_df = df.tail(best_tws).reset_index(drop=True)
    next_n_rows = df.iloc[similar_df_start_index: similar_df_end_index + future_periods + 1]
    next_n_rows = next_n_rows.reset_index(drop=True)

    main_df = pd.merge(next_n_rows, target_df, left_index=True, right_index=True, how='outer')
    main_df = main_df.rename(columns={
        f'{target_column}_x':'Similar',
        f'{target_column}_y':'Target'
    })

    ts = main_df['Target'] / main_df['Similar']
    stats = {
        'min': np.min(ts),
        'mean': np.mean(ts),
        'max': np.max(ts),
        'percentile_25': np.percentile(ts.dropna(), 25),
        'percentile_75': np.percentile(ts.dropna(), 75)
    }

    for stat_name, stat_value in stats.items():
        main_df[f'{stat_name.capitalize()}_Target_Similar'] = np.where(
            main_df['Target'].isnull(),
            main_df['Similar'] * stat_value,
            main_df['Target']
        )

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(main_df.index, main_df['Similar'], color='tab:blue', label='Similar')
    ax1.set_xlabel('')
    ax1.set_ylabel('Similar', color='tab:blue')
    ax1.axvline(x=best_tws - 1, color='gray', linestyle='--')

    ax2 = ax1.twinx()
    ax2.plot(main_df.index, main_df['Mean_Target_Similar'], color='tab:red', label='Target')
    ax2.fill_between(main_df.index, main_df['Min_Target_Similar'], main_df['Max_Target_Similar'], color='gray', alpha=0.3, label='Min-Max Range')
    ax2.fill_between(main_df.index, main_df['Percentile_25_Target_Similar'], main_df['Percentile_75_Target_Similar'], color='orange', alpha=0.3, label='25-75 Percentile Range')
    ax2.set_ylabel('Target', color='tab:red')

    title = 'Dynamic Time Warping' if distance == 'dtw' else 'Euclidean'
    plt.title(
        f"Possible {future_periods}-Period Movement of ${ticker.split("-")[0]} Based on Similar Past Movements [Interval: {interval}] ({title} Used)",
        fontsize=14
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
    ax2.legend(lines + lines2, labels + labels2, loc='upper left', fontsize='small')

    plt.grid(True)
    plt.show()

    return main_df

ticker='BTC-USD'
start_date='2014-01-01'
end_date='2024-03-07'
interval='1d'

df = yf.download(
    tickers=ticker,
    start=start_date,
    end=end_date,
    interval=interval,
    progress=False
)

distance='euclidean'
target_window_size=np.arange(20,105,5)

result_df = historical_similar_movements_dtw(
    df,
    distance=distance,
    target_window_size=target_window_size
)