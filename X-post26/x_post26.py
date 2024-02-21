import yfinance as yf
import numpy as np
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')

def find_local_extrema(df, extrema_type='Low', show_plot=True):

    opposite_extrema_type = 'High' if extrema_type == 'Low' else 'Low'

    if extrema_type == 'Low':
        df['local_extrema'] = df[extrema_type][
            (df[extrema_type].shift(1) > df[extrema_type]) &
            (df[extrema_type].shift(-1) > df[extrema_type])
        ]
    else:
        df['local_extrema'] = df[extrema_type][
            (df[extrema_type].shift(1) < df[extrema_type]) &
            (df[extrema_type].shift(-1) < df[extrema_type])
        ]

    df['local_type'] = np.where(~df['local_extrema'].isna(), extrema_type, '')

    for i in range(len(df)):
        if df['local_type'].iloc[i] == extrema_type:
            start_index = i + 1
            end_index = None
            j = i + 2

            while j < len(df):
                if df['local_type'].iloc[j] == '':
                    j += 1
                else:
                    end_index = j
                    opposite_extreme = df[opposite_extrema_type][start_index:end_index]
                    extreme_value = opposite_extreme.max() if opposite_extrema_type == 'High' else opposite_extreme.min()
                    extreme_index = opposite_extreme.idxmax() if opposite_extrema_type == 'High' else opposite_extreme.idxmin()
                    df.at[extreme_index, 'local_extrema'] = extreme_value
                    df.at[extreme_index, 'local_type'] = opposite_extrema_type
                    i = j
                    break

    first_index = df[df['local_type'] == extrema_type].index[0]
    first_df_index = df.index[0]

    if first_df_index != first_index:
        if extrema_type == 'Low':
            opposite_value_first = df[opposite_extrema_type][first_df_index:first_index][:-1].max()
            opposite_value_first_index = df[opposite_extrema_type][first_df_index:first_index][:-1].idxmax()
        else:
            opposite_value_first = df[opposite_extrema_type][first_df_index:first_index][:-1].min()
            opposite_value_first_index = df[opposite_extrema_type][first_df_index:first_index][:-1].idxmin()

        df.at[opposite_value_first_index, 'local_extrema'] = opposite_value_first
        df.at[opposite_value_first_index, 'local_type'] = opposite_extrema_type

    last_index = df[df['local_type'] == extrema_type].index[-1]
    last_df_index = df.index[-1]

    if last_index != last_df_index:
        if extrema_type == 'Low':
            opposite_value_last = df[opposite_extrema_type][last_index:last_df_index][1:].max()
            opposite_value_last_index = df[opposite_extrema_type][last_index:last_df_index][1:].idxmax()
        else:
            opposite_value_last = df[opposite_extrema_type][last_index:last_df_index][1:].min()
            opposite_value_last_index = df[opposite_extrema_type][last_index:last_df_index][1:].idxmin()

        df.at[opposite_value_last_index, 'local_extrema'] = opposite_value_last
        df.at[opposite_value_last_index, 'local_type'] = opposite_extrema_type

    extrema_df = df.dropna()
    extrema_df['local_extrema_pct'] = extrema_df['local_extrema'].pct_change() * 100

    if show_plot:
        candlestick = go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Candlesticks'
        )

        local_extrema_points = go.Scatter(
            x=extrema_df.index,
            y=extrema_df['local_extrema'],
            mode='markers+text',
            marker=dict(color=np.where(extrema_df['local_type'] == 'High', 'blue', 'red'), size=9, opacity=0.5),
            text=np.where(extrema_df.index == extrema_df.index[0], '', (extrema_df['local_extrema_pct'].round(1)).astype(str) + '%'),
            textposition=np.where(extrema_df['local_type'] == 'High', 'top center', 'bottom center'),
            textfont=dict(size=9),
            name='Local Extrema'
        )

        lines = go.Scatter(
            x=extrema_df.index,
            y=extrema_df['local_extrema'],
            mode='lines',
            line=dict(color='black', width=2),
            name='Lines'
        )

        layout = go.Layout(
            title=dict(
                text=f'${ticker.split("-")[0]}, Local Peaks and Troughs',
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(title='', rangeslider=dict(visible=False)),
            yaxis=dict(title='Price'),
            width=1000,
            height=650,
            showlegend=False
        )

        fig = go.Figure(data=[candlestick, local_extrema_points, lines], layout=layout)
        fig.show()

    return extrema_df

ticker = 'BTC-USD'
start_date = '2023-01-01'
interval = '1wk'

df = yf.download(
    tickers=ticker,
    start=start_date,
    interval=interval,
    progress=False
)[['Open', 'High', 'Low', 'Adj Close']]
df = df.rename(columns={'Adj Close': 'Close'})
df = df.drop(df.index[-1])

result_df = find_local_extrema(df, extrema_type='High')