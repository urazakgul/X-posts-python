import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from requests_html import HTMLSession
import plotly.graph_objs as go
import warnings
warnings.filterwarnings("ignore")

def triangle_patterns(
        tickers=['BTC-USD','ETH-USD'],
        num_recent_candles=50,
        upper_body_threshold=30,
        lower_body_threshold=30,
        upper_body_tolerance=75,
        lower_body_tolerance=75,
        symmetrical_triangle_upper_threshold_negative=-30,
        symmetrical_triangle_lower_threshold_positive=30,
        ascending_triangle_upper_threshold_negative=-2,
        ascending_triangle_upper_threshold_positive=2,
        ascending_triangle_lower_threshold_positive=30,
        descending_triangle_upper_threshold_negative=-30,
        descending_triangle_lower_threshold_negative=-2,
        descending_triangle_lower_threshold_positive=2,
        folder_name='triangle_imgs'
):

    tickers = [tickers] if not isinstance(tickers, list) else tickers
    num_recent_candles = [num_recent_candles] if not isinstance(num_recent_candles, list) else num_recent_candles

    for ticker in tickers:

        for nrc in num_recent_candles:

            try:

                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=nrc)).strftime('%Y-%m-%d')

                recent_data = yf.download(
                    tickers=ticker, start=start_date, end=end_date, interval='1d', progress=False
                )[['Open','High','Low','Adj Close']].rename(columns={'Adj Close':'Close'})

                recent_data = recent_data.tail(nrc)

                recent_data['UL'] = float('nan')
                recent_data['UL_Type'] = ''

                max_upper_index = recent_data[['Open', 'Close']].max(axis=1).idxmax()
                max_value_col = 'Open' if recent_data.loc[max_upper_index, 'Open'] > recent_data.loc[max_upper_index, 'Close'] else 'Close'
                recent_data.loc[max_upper_index, ['UL', 'UL_Type']] = [recent_data.loc[max_upper_index, max_value_col], 'Upper OC']

                min_lower_index = recent_data[['Open', 'Close']].min(axis=1).idxmin()
                min_value_col = 'Open' if recent_data.loc[min_lower_index, 'Open'] < recent_data.loc[min_lower_index, 'Close'] else 'Close'
                recent_data.loc[min_lower_index, ['UL', 'UL_Type']] = [recent_data.loc[min_lower_index, min_value_col], 'Lower OC']

                upper_filtered = recent_data[recent_data['UL_Type'] == 'Upper OC']
                if recent_data.iloc[-1]['UL_Type'] not in ['Upper OC']:
                    upper_filtered = pd.concat([upper_filtered, recent_data.iloc[[-1]]])
                    upper_filtered.loc[upper_filtered.index[-1], 'UL'] = max(upper_filtered.loc[upper_filtered.index[-1], 'Open'], upper_filtered.loc[upper_filtered.index[-1], 'Close'])
                    upper_filtered.loc[upper_filtered.index[-1], 'UL_Type'] = 'Upper OC'
                upper_filtered['HPC'] = upper_filtered['UL'].pct_change() * 100

                lower_filtered = recent_data[recent_data['UL_Type'] == 'Lower OC']
                if recent_data.iloc[-1]['UL_Type'] not in ['Lower OC']:
                    lower_filtered = pd.concat([lower_filtered, recent_data.iloc[[-1]]])
                    lower_filtered.loc[lower_filtered.index[-1], 'UL'] = min(lower_filtered.loc[lower_filtered.index[-1], 'Open'], lower_filtered.loc[lower_filtered.index[-1], 'Close'])
                    lower_filtered.loc[lower_filtered.index[-1], 'UL_Type'] = 'Lower OC'
                lower_filtered['LPC'] = lower_filtered['UL'].pct_change() * 100

                recent_data_dates = pd.to_datetime(recent_data.index.tolist())

                upper_first_date = upper_filtered.index[0]
                upper_last_date = upper_filtered.index[-1]
                upper_days_between = recent_data_dates.get_loc(upper_last_date) - recent_data_dates.get_loc(upper_first_date) - 1

                upper_first_value = upper_filtered['UL'].iloc[0]
                upper_last_value = upper_filtered['UL'].iloc[-1]

                upper_step_size = (upper_last_value - upper_first_value) / (upper_days_between + 1)

                upper_values = [upper_first_value + i * upper_step_size for i in range(1, upper_days_between + 1)]
                upper_dates = recent_data_dates[(recent_data_dates > upper_first_date) & (recent_data_dates < upper_last_date)]

                upper_df = pd.DataFrame({
                    'Date': upper_dates,
                    'Upper_Values': upper_values
                })
                upper_df = pd.concat([
                    pd.DataFrame({'Date': [upper_first_date], 'Upper_Values': [upper_first_value]}),
                    upper_df,
                    pd.DataFrame({'Date': [upper_last_date], 'Upper_Values': [upper_last_value]})
                ], ignore_index=True)
                upper_df = upper_df.set_index('Date')

                upper_recent_data = recent_data[['Open','High','Low','Close']]
                upper_merged_df = upper_recent_data.join(upper_df, how='right')
                upper_merged_df['Upper_Check'] = 0
                for index, row in upper_merged_df.iterrows():
                    if row['Open'] < row['Close'] and row['Close'] > row['Upper_Values'] and row['Open'] < row['Upper_Values']:
                        first_diff = abs(row['Open'] - row['Close'])
                        second_diff = abs(row['Upper_Values'] - row['Close'])
                        upper_merged_df.at[index, 'Upper_Check'] = (second_diff / first_diff) * 100
                    elif row['Open'] > row['Close'] and row['Open'] > row['Upper_Values'] and row['Close'] < row['Upper_Values']:
                        first_diff = abs(row['Open'] - row['Close'])
                        second_diff = abs(row['Upper_Values'] - row['Open'])
                        upper_merged_df.at[index, 'Upper_Check'] = (second_diff / first_diff) * 100
                    elif row['Open'] <= row['Upper_Values'] and row['Close'] <= row['Upper_Values']:
                        upper_merged_df.at[index, 'Upper_Check'] = 0
                    else:
                        upper_merged_df.at[index, 'Upper_Check'] = 100

                upper_merged_df['Upper_Check_N'] = ''
                for index, row in upper_merged_df.iterrows():
                    if row['Upper_Check'] <= upper_body_threshold:
                        upper_merged_df.at[index, 'Upper_Check_N'] = 'OK'
                    else:
                        upper_merged_df.at[index, 'Upper_Check_N'] = ''

                upper_ok_count = (upper_merged_df['Upper_Check_N'] == 'OK').sum()
                upper_total_rows = upper_merged_df.shape[0]
                upper_ok_ratio = upper_ok_count / upper_total_rows * 100

                lower_first_date = lower_filtered.index[0]
                lower_last_date = lower_filtered.index[-1]
                lower_days_between = recent_data_dates.get_loc(lower_last_date) - recent_data_dates.get_loc(lower_first_date) - 1

                lower_first_value = lower_filtered['UL'].iloc[0]
                lower_last_value = lower_filtered['UL'].iloc[-1]

                lower_step_size = (lower_last_value - lower_first_value) / (lower_days_between + 1)

                lower_values = [lower_first_value + i * lower_step_size for i in range(1, lower_days_between + 1)]
                lower_dates = recent_data_dates[(recent_data_dates > lower_first_date) & (recent_data_dates < lower_last_date)]

                lower_df = pd.DataFrame({
                    'Date': lower_dates,
                    'Lower_Values': lower_values
                })
                lower_df = pd.concat([
                    pd.DataFrame({'Date': [lower_first_date], 'Lower_Values': [lower_first_value]}),
                    lower_df,
                    pd.DataFrame({'Date': [lower_last_date], 'Lower_Values': [lower_last_value]})
                ], ignore_index=True)
                lower_df = lower_df.set_index('Date')

                lower_recent_data = recent_data[['Open','High','Low','Close']]
                lower_merged_df = lower_recent_data.join(lower_df, how='right')
                lower_merged_df['Lower_Check'] = 0
                for index, row in lower_merged_df.iterrows():
                    if row['Open'] < row['Close'] and row['Open'] < row['Lower_Values'] and row['Close'] > row['Lower_Values']:
                        first_diff = abs(row['Open'] - row['Close'])
                        second_diff = abs(row['Lower_Values'] - row['Open'])
                        lower_merged_df.at[index, 'Lower_Check'] = (second_diff / first_diff) * 100
                    elif row['Open'] > row['Close'] and row['Close'] < row['Lower_Values'] and row['Open'] > row['Lower_Values']:
                        first_diff = abs(row['Open'] - row['Close'])
                        second_diff = abs(row['Lower_Values'] - row['Close'])
                        lower_merged_df.at[index, 'Lower_Check'] = (second_diff / first_diff) * 100
                    elif row['Open'] >= row['Lower_Values'] and row['Close'] >= row['Lower_Values']:
                        lower_merged_df.at[index, 'Lower_Check'] = 0
                    else:
                        lower_merged_df.at[index, 'Lower_Check'] = 100

                lower_merged_df['Lower_Check_N'] = ''
                for index, row in lower_merged_df.iterrows():
                    if row['Lower_Check'] <= lower_body_threshold:
                        lower_merged_df.at[index, 'Lower_Check_N'] = 'OK'
                    else:
                        lower_merged_df.at[index, 'Lower_Check_N'] = ''

                lower_ok_count = (lower_merged_df['Lower_Check_N'] == 'OK').sum()
                lower_total_rows = lower_merged_df.shape[0]
                lower_ok_ratio = lower_ok_count / lower_total_rows * 100

                num_rows = recent_data.shape[0]
                middle_index = num_rows // 2
                middle_date = recent_data.index[middle_index]

                print(f'''
                Ticker: {ticker}
                Upper Max: {upper_filtered['HPC'].max():.1f}
                Lower Min: {lower_filtered['LPC'].min():.1f}
                ''')


                triangle_type = ''

                if upper_first_date <= middle_date and lower_first_date <= middle_date:
                    if upper_ok_ratio >= upper_body_tolerance and lower_ok_ratio >= lower_body_tolerance:
                        if upper_filtered['HPC'].max() <= symmetrical_triangle_upper_threshold_negative and lower_filtered['LPC'].min() >= symmetrical_triangle_lower_threshold_positive:
                            triangle_type = 'Symmetrical'
                        elif ascending_triangle_upper_threshold_negative <= upper_filtered['HPC'].max() <= ascending_triangle_upper_threshold_positive and lower_filtered['LPC'].min() >= ascending_triangle_lower_threshold_positive:
                            triangle_type = 'Ascending'
                        elif descending_triangle_lower_threshold_negative <= lower_filtered['LPC'].min() <= descending_triangle_lower_threshold_positive and upper_filtered['HPC'].max() <= descending_triangle_upper_threshold_negative:
                            triangle_type = 'Descending'
                        else:
                            triangle_type = 'No triangle detected'

                trace = go.Candlestick(
                    x=recent_data.index,
                    open=recent_data['Open'],
                    high=recent_data['High'],
                    low=recent_data['Low'],
                    close=recent_data['Close'],
                    name=f'{ticker}'
                )

                upper_scatter = go.Scatter(
                    x=upper_filtered.index,
                    y=upper_filtered['UL'],
                    mode='markers',
                    marker=dict(color='green', size=10, symbol='circle'),
                    name='Upper'
                )

                upper_line = go.Scatter(
                    x=upper_merged_df.index,
                    y=upper_merged_df['Upper_Values'],
                    mode='lines',
                    line=dict(color='green', width=2),
                    name='Upper Line'
                )

                lower_scatter = go.Scatter(
                    x=lower_filtered.index,
                    y=lower_filtered['UL'],
                    mode='markers',
                    marker=dict(color='red', size=10, symbol='circle'),
                    name='Lower'
                )

                lower_line = go.Scatter(
                    x=lower_merged_df.index,
                    y=lower_merged_df['Lower_Values'],
                    mode='lines',
                    line=dict(color='red', width=2),
                    name='Lower Line'
                )

                layout = go.Layout(
                    title=f'{triangle_type} triangle has been detected on {ticker}',
                    xaxis=dict(title='', rangeslider=dict(visible=False), type='category'),
                    yaxis=dict(title='Closing Price'),
                    hovermode='closest',
                    showlegend=True,
                    width=800,
                    height=600
                )

                fig = go.Figure(
                    data=[
                        trace,
                        upper_scatter,
                        upper_line,
                        lower_scatter,
                        lower_line
                    ],
                    layout=layout
                )

                # fig.show()

                if triangle_type not in 'No triangle detected':
                    fig.write_image(f"./{folder_name}/{ticker}_{triangle_type.lower()}_triangle_{start_date}_{end_date}_{nrc}.png")

            except Exception as e:
                print(f"Failed download: [{ticker}]: {e}")
                continue

def fetch_crypto_data(num_pages=1):
    tickers = []

    for page in range(1, num_pages + 1):
        offset = (page - 1) * 100
        session = HTMLSession()
        res = session.get(f"https://finance.yahoo.com/crypto?count=100&offset={offset}")

        if res.status_code == 200:
            tables = pd.read_html(res.html.raw_html)
            df = tables[0].copy()
            tickers.extend(df.Symbol.tolist())
        else:
            print(f"Error: Failed to fetch data from page {page}")

    return tickers

tickers = fetch_crypto_data(num_pages=1)[:50]
num_recent_candles = [50,60,70]

triangle_patterns(
    tickers=tickers,
    num_recent_candles=num_recent_candles
)