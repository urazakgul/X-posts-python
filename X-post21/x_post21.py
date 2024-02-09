import yfinance as yf
from datetime import timedelta
import pandas as pd
import numpy as np

def engulfing_patterns(data, body_ratio=0.3, ema_length=7):
    df_pattern = data.copy()
    df_pattern[f'EMA_{ema_length}'] = df_pattern['Close'].ewm(span=ema_length, adjust=False, min_periods=ema_length).mean()

    df_pattern.loc[:, 'Bearish_Engulfing'] = False
    df_pattern.loc[:, 'Bullish_Engulfing'] = False

    df_pattern = df_pattern.dropna()

    for i in range(1, len(df_pattern)):
        current_open = df_pattern['Open'].iloc[i]
        current_high = df_pattern['High'].iloc[i]
        current_low = df_pattern['Low'].iloc[i]
        current_close = df_pattern['Close'].iloc[i]

        previous_open = df_pattern['Open'].iloc[i - 1]
        previous_high = df_pattern['High'].iloc[i - 1]
        previous_low = df_pattern['Low'].iloc[i - 1]
        previous_close = df_pattern['Close'].iloc[i - 1]

        current_candle_body = abs((current_open - current_close)) >= (current_high - current_low) * body_ratio

        current_ema = df_pattern[f'EMA_{ema_length}'].iloc[i]

        current_close_below_ema = current_close < current_ema
        current_close_above_ema = current_close > current_ema

        is_bearish_engulfing = (
            (previous_open < previous_close) and
            (current_open > current_close) and
            (current_open > previous_high) and
            (current_close < previous_open) and
            current_candle_body and
            current_close_below_ema
        )

        is_bullish_engulfing = (
            (previous_open > previous_close) and
            (current_open < current_close) and
            (current_open < previous_close) and
            (current_close > previous_high) and
            current_candle_body and
            current_close_above_ema
        )

        if is_bearish_engulfing:
            df_pattern.loc[df_pattern.index[i], 'Bearish_Engulfing'] = True
        else:
            df_pattern.loc[df_pattern.index[i], 'Bearish_Engulfing'] = False

        if is_bullish_engulfing:
            df_pattern.loc[df_pattern.index[i], 'Bullish_Engulfing'] = True
        else:
            df_pattern.loc[df_pattern.index[i], 'Bullish_Engulfing'] = False

    return df_pattern

def avg_engulfing_price_change(days_after_engulfing=3):

    tickers = [
        'BTC-USD',
        'ETH-USD',
        'BNB-USD',
        'SOL-USD',
        'XRP-USD',
        'ADA-USD',
        'AVAX-USD',
        'DOGE-USD',
        'LINK-USD',
        'DOT-USD',
        'MATIC-USD',
        'SHIB-USD',
        'LTC-USD',
        'ATOM-USD',
        'XLM-USD',
        'INJ-USD',
        'OKB-USD'
    ]

    engulfing_changes_all = []

    for ticker in tickers:
        df = yf.download(
            tickers=ticker,
            start='2014-01-01',
            interval='1d',
            progress=False
        )[['Open', 'High', 'Low', 'Close']]

        df_pattern = engulfing_patterns(df, body_ratio=0.3, ema_length=7)

        engulfing_changes_list = []

        for num_days in range(1, days_after_engulfing + 1):
            bearish_engulfing_indices = df_pattern[df_pattern['Bearish_Engulfing']].index
            bullish_engulfing_indices = df_pattern[df_pattern['Bullish_Engulfing']].index

            for idx, engulfing_type in zip([bearish_engulfing_indices, bullish_engulfing_indices], ['Bearish', 'Bullish']):
                for idx_val in idx:
                    next_day_date = idx_val + timedelta(days=num_days)
                    if next_day_date > df_pattern.index[-1]:
                        continue
                    next_day_close = df.loc[next_day_date, 'Close']
                    current_close = df.loc[idx_val, 'Close']
                    price_change = ((next_day_close - current_close) / current_close) * 100

                    engulfing_changes_list.append({
                        'Ticker': ticker,
                        'Engulfing_Type': engulfing_type,
                        'Price_Change': price_change,
                        'Days_Ahead': num_days,
                        'Date': idx_val
                    })

        engulfing_changes_all.extend(engulfing_changes_list)

    df_engulfing_changes_all = pd.DataFrame(engulfing_changes_all)

    avg_price_change = df_engulfing_changes_all.groupby(['Ticker', 'Engulfing_Type', 'Days_Ahead'])['Price_Change'].mean().reset_index()

    avg_price_change['Signal'] = avg_price_change.apply(
        lambda row: 'GOOD' if (row['Engulfing_Type'] == 'Bearish' and row['Price_Change'] < 0) or (row['Engulfing_Type'] == 'Bullish' and row['Price_Change'] > 0) else 'BAD', axis=1
    )

    return avg_price_change

def find_best_parameters():
    best_params = {'body_ratio': None, 'ema_length': None, 'good_ratio': 0}

    body_ratio_range = np.arange(0.3, 0.8, 0.1)
    ema_length_range = [7, 14, 21]

    for body_ratio in body_ratio_range:
        for ema_length in ema_length_range:
            print(f'Body Ratio: {body_ratio:.1f} and EMA Length: {ema_length:.1f}')
            avg_change_df = avg_engulfing_price_change(days_after_engulfing=3)
            good_count = len(avg_change_df[(avg_change_df['Signal'] == 'GOOD')])
            total_count = len(avg_change_df)

            if total_count != 0:
                current_good_ratio = good_count / total_count
                if current_good_ratio > best_params['good_ratio']:
                    best_params['good_ratio'] = current_good_ratio
                    best_params['body_ratio'] = body_ratio
                    best_params['ema_length'] = ema_length

    return best_params

best_params = find_best_parameters()

print("Best parameters:")
print("body_ratio: {:.1f}".format(best_params['body_ratio']))
print("ema_length: {:.1f}".format(best_params['ema_length']))
print("GOOD ratio: {:.1%}".format(best_params['good_ratio']))