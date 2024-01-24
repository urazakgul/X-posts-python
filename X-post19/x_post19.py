import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

def find_top_patterns(df, target_pattern_length=3, future_pattern_length=1, top_n_patterns=2, include_last_bar=False):

    df['RG'] = np.where(df['Open'] > df['Close'], 'R', 'G')
    df = df.reset_index(drop=True)

    if not include_last_bar:
        df = df.drop(df.index[-1])

    target_pattern = df['RG'].tail(target_pattern_length).tolist()

    df = df.iloc[:-target_pattern_length]

    results = []

    for i in range(0, len(df)):
        chunk = df.iloc[i:i+len(target_pattern)]
        if len(chunk) != target_pattern_length:
            break
        else:
            if chunk['RG'].tolist() == target_pattern:
                start_row = chunk.index[0]
                end_row = chunk.index[-1]
                results.append({
                    'start_row': start_row,
                    'end_row': end_row
                })

    master_df = pd.DataFrame(columns=['Group', 'Pattern'])
    group_num = 1

    for result in results:
        end_row = result['end_row']

        if (end_row + future_pattern_length) <= df.index[-1]:
            selected_rows = df.loc[end_row + 1:end_row + future_pattern_length, ['RG']].copy()
            merged_rg = ''.join(selected_rows['RG'])
            sub_df = pd.DataFrame({
                'Group': [group_num],
                'Pattern': [merged_rg]
            })
            master_df = pd.concat([master_df, sub_df], ignore_index=True)

            group_num += 1

    pattern_counts = master_df['Pattern'].value_counts()
    total_patterns = len(master_df)
    pattern_proportions = pattern_counts / total_patterns
    result_df = pd.DataFrame({
        'Pattern': pattern_counts.index,
        'Occurrences': pattern_counts.values,
        'Proportion': pattern_proportions.values
    })
    result_df = result_df.sort_values(by='Proportion', ascending=False)

    top_patterns = result_df.head(top_n_patterns)

    print(f"Top {top_n_patterns} Patterns with Proportions for Target Pattern: {''.join(target_pattern)}")
    print("---------------------------------")
    for index, row in top_patterns.iterrows():
        pattern = row['Pattern']
        proportion = row['Proportion'] * 100
        print(f"Pattern: {pattern}, Proportion: {proportion:.2f}%")

ticker = 'INJ-USD'
start_date = (datetime.now() - timedelta(days=729)).strftime('%Y-%m-%d')
interval = '1h'

df = yf.download(
    tickers=ticker,
    start=start_date,
    interval=interval,
    progress=False
)[['Open', 'Close']]

target_pattern_length = 5
future_pattern_length = 2
top_n_patterns = 3
include_last_bar=False

find_top_patterns(
    df,
    target_pattern_length,
    future_pattern_length,
    top_n_patterns,
    include_last_bar
)