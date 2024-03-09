import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from dtaidistance import dtw
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# If history repeats itself...

def historical_similar_movements(
        df,
        normalize=True,
        distance=['euclidean','dtw'],
        target_column='Close',
        target_window_size=[20,30,40,50],
        future_periods=5,
        plot=True
):

    df = df[[target_column]]

    distance_score = []

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

            if 'dtw' in distance:

                dtw_distance = dtw.distance(
                    target_df[target_column].values,
                    sub_rolling_df[target_column].values
                )

                distance_score.append({
                    'Start Date': start_date,
                    'End Date': end_date,
                    'Score': dtw_distance,
                    'TWS': tws,
                    'Distance': 'DTW'
                })

            if 'euclidean' in distance:

                euclidean_distance = np.linalg.norm(
                    target_df[target_column].values - sub_rolling_df[target_column].values
                )

                distance_score.append({
                    'Start Date': start_date,
                    'End Date': end_date,
                    'Score': euclidean_distance,
                    'TWS': tws,
                    'Distance': 'Euclidean'
                })

    distance_scores = pd.DataFrame(distance_score)

    min_score_row = distance_scores.sort_values('Score').iloc[0]
    best_start_date = min_score_row['Start Date']
    best_end_date = min_score_row['End Date']
    best_tws = min_score_row['TWS']
    best_distance = min_score_row['Distance']

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

    if plot:

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.plot(main_df.index, main_df['Similar'], color='tab:blue', label='Similar')
        ax1.set_xlabel('')
        ax1.set_ylabel('Similar', color='tab:blue')
        ax1.axvline(x=best_tws - 1, color='gray', linestyle='--')

        ax2 = ax1.twinx()
        ax2.plot(main_df.index, main_df['Mean_Target_Similar'], color='tab:red', label='Target')
        ax2.fill_between(
            main_df.index,
            main_df['Min_Target_Similar'],
            main_df['Max_Target_Similar'],
            color='gray',
            alpha=0.3,
            label='Min-Max Range'
        )
        ax2.fill_between(
            main_df.index,
            main_df['Percentile_25_Target_Similar'],
            main_df['Percentile_75_Target_Similar'],
            color='orange',
            alpha=0.3,
            label='25-75 Percentile Range'
        )
        ax2.set_ylabel('Target', color='tab:red')

        plt.title(
            f"Possible {future_periods}-Period Movement of ${ticker.split("-")[0]} Based on Similar Past Movements [Interval: {interval}] ({best_distance} Used)",
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

# ticker='BTC-USD'
# start_date='2014-01-01'
# end_date='2024-03-07'
# interval='1d'

# df = yf.download(
#     tickers=ticker,
#     start=start_date,
#     end=end_date,
#     interval=interval,
#     progress=False
# )

# main_df = historical_similar_movements(
#     df,
#     target_window_size=np.arange(20,105,5)
# )

### BACKTESTING ###

def historical_similar_movements_backtest(
        df,
        start_date='2014-07-19',
        min_end_date='2023-01-01',
        max_end_date='2024-02-01',
        random_date_size=100,
        metric='rmse',
        backtest_plot=True,
        normalize=True,
        distance=['euclidean','dtw'],
        target_column='Close',
        target_window_size=[20,30,40,50],
        future_periods=5,
        plot=False
):

    # The metric parameter must be either 'rmse', 'mae', or 'mse'

    user_min_end_date = datetime.strptime(min_end_date, '%Y-%m-%d')
    user_max_end_date = datetime.strptime(max_end_date, '%Y-%m-%d')
    date_list = []
    while len(date_list) < random_date_size:
        random_date = user_min_end_date + timedelta(days=random.randint(0, (user_max_end_date - user_min_end_date).days))
        if random_date not in date_list:
            date_list.append(random_date)

    forecast_df = pd.DataFrame()

    for dl in range(len(date_list)):

        if date_list[dl] in df.index:

            test_df = df[(df.index >= start_date) & (df.index <= date_list[dl])]

            test_result_df = historical_similar_movements(
                test_df,
                normalize=normalize,
                distance=distance,
                target_column=target_column,
                target_window_size=target_window_size,
                plot=plot
            )

            forecast = (test_result_df[test_result_df['Target'].isnull()]['Mean_Target_Similar']).values
            actual = (df[df.index > date_list[dl]].head(future_periods)['Close']).values

            actual_forecast = pd.DataFrame({
                'End Date': [date_list[dl].strftime('%Y-%m-%d')] * future_periods,
                'Forecast': forecast,
                'Actual': actual
            })

            forecast_df = pd.concat([forecast_df, actual_forecast], ignore_index=True)

            percentage = (dl + 1) / len(date_list) * 100
            print(f"{percentage:.0f}% completed")

    error_list = []
    metric_functions = {
        'rmse': lambda actual, forecast: np.sqrt(mean_squared_error(actual, forecast)),
        'mae': lambda actual, forecast: mean_absolute_error(actual, forecast),
        'mse': lambda actual, forecast: mean_squared_error(actual, forecast)
    }

    for end_date, group in forecast_df.groupby('End Date'):
        metric_value = metric_functions[metric](group['Actual'], group['Forecast'])
        error_list.append({'End Date': end_date, f'{metric.upper()}': metric_value})

    metric_df = pd.DataFrame(error_list)

    if backtest_plot:

        mean_value = np.mean(metric_df[f'{metric.upper()}'])
        median_value = np.median(metric_df[f'{metric.upper()}'])
        ax = sns.displot(
            metric_df[f'{metric.upper()}'],
            kind="kde",
            fill=True,
            cut=0,
            height=6,
            aspect=2,
            rug=True
        )
        ax.set(yticks=[])
        ax.ax.axvline(mean_value, color='r', linestyle='--')
        ax.ax.axvline(median_value, color='b', linestyle='--')
        ax.ax.text(
            0.95,
            0.95,
            f'Average {metric.upper()}: {mean_value:.0f}',
            color='r',
            ha='right',
            va='top',
            transform=ax.ax.transAxes
        )
        ax.ax.text(
            0.95,
            0.90,
            f'Median {metric.upper()}: {median_value:.0f}',
            color='b',
            ha='right',
            va='top',
            transform=ax.ax.transAxes
        )

        fig, ax2 = plt.subplots(figsize=(12, 6))
        sns.scatterplot(
            data=forecast_df,
            x='Forecast',
            y='Actual',
            color='blue',
            ax=ax2
        )
        sns.regplot(
            data=forecast_df,
            x='Forecast',
            y='Actual',
            scatter=False,
            color='red',
            ax=ax2
        )

        plt.show()

    return forecast_df, metric_df

ticker='BTC-USD'
start_date='2014-07-19'
end_date='2024-03-09'
interval='1d'

df = yf.download(
    tickers=ticker,
    start=start_date,
    end=end_date,
    interval=interval,
    progress=False
)

forecast_df, metric_df = historical_similar_movements_backtest(
    df,
    start_date=start_date,
    min_end_date='2022-01-01',
    max_end_date='2024-02-29',
    random_date_size=50,
    metric='rmse',
    backtest_plot=True,
    normalize=True,
    distance=['euclidean','dtw'],
    target_column='Close',
    target_window_size=np.arange(20,60,10),
    future_periods=5,
    plot=False
)