import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

def regression_repainting(df, logarithmic=True, initial_sample_size=50, significance_levels=[0.01, 0.05]):

    df = df[['Open', 'High', 'Low', 'Adj Close']]
    df = df.rename(columns={'Adj Close': 'Close'})

    if logarithmic:
            df = np.log(df)

    df = df.reset_index()

    play_button = {
        'args': [
            None,
            {
                'frame': {'duration': 300, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 50,'easing': 'quadratic-in-out'}
            }
        ],
        'label': 'Play',
        'method': 'animate'
    }

    pause_button = {
        'args': [
            [None],
            {
                'frame': {'duration': 0, 'redraw': False},
                'mode': 'immediate',
                'transition': {'duration': 0}
            }
        ],
        'label': 'Pause',
        'method': 'animate'
    }

    initial_plot = go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=f'{ticker}'
    )

    frames = []
    signal_df = pd.DataFrame(columns=['Date', 'Close', 'Signal'])

    for i in range(initial_sample_size, len(df)):
        X = np.arange(1, (i + 1), 1)
        X = sm.add_constant(X)
        y = df['Close'][:i]
        model = sm.OLS(y, X).fit()

        start_index = 0
        end_index = i - 1

        for significance_level in significance_levels:
            predictions = model.get_prediction(exog=X).summary_frame(alpha=significance_level)
            df.loc[df.index[start_index:end_index], f'Fitted_{significance_level}'] = predictions['mean']
            df.loc[df.index[start_index:end_index], f'Lower_{significance_level}'] = predictions['obs_ci_lower']
            df.loc[df.index[start_index:end_index], f'Upper_{significance_level}'] = predictions['obs_ci_upper']

        for j in range(start_index, end_index + 1):
            close_price = df.loc[df.index[j], 'Close']
            max_lower = max(df.loc[df.index[j], [f'Lower_{significance_level}' for significance_level in significance_levels]])
            min_upper = min(df.loc[df.index[j], [f'Upper_{significance_level}' for significance_level in significance_levels]])

            if close_price < max_lower:
                signal_data = pd.DataFrame({'Date': [df['Date'].iloc[j]], 'Close': [close_price], 'Signal': ['Buy']})
                signal_df = pd.concat([signal_df, signal_data], ignore_index=True)
            elif close_price > min_upper:
                signal_data = pd.DataFrame({'Date': [df['Date'].iloc[j]], 'Close': [close_price], 'Signal': ['Sell']})
                signal_df = pd.concat([signal_df, signal_data], ignore_index=True)

        reg_lines = []
        for significance_level in significance_levels:
            reg_lines.append(
                go.Scatter(
                    x=df['Date'].iloc[:i],
                    y=df[f'Fitted_{significance_level}'],
                    mode='lines',
                    line=dict(color='blue', width=1),
                    name=f'Fitted {significance_level}'
                )
            )
            reg_lines.append(
                go.Scatter(
                    x=df['Date'].iloc[:i],
                    y=df[f'Lower_{significance_level}'],
                    mode='lines',
                    line=dict(color='green', width=1, dash='dash'),
                    name=f'Lower {significance_level}'
                )
            )
            reg_lines.append(
                go.Scatter(
                    x=df['Date'].iloc[:i],
                    y=df[f'Upper_{significance_level}'],
                    mode='lines',
                    line=dict(color='red', width=1, dash='dash'),
                    name=f'Upper {significance_level}'
                )
            )

        upper_columns = [col for col in df.columns if col.startswith('Upper')]
        upper_max_col = max(upper_columns, key=lambda x: df[x].max())
        upper_min_col = min(upper_columns, key=lambda x: df[x].min())

        reg_lines.append(
             go.Scatter(
                  x=df['Date'],
                  y=df[upper_max_col],
                  line=dict(color='rgba(0,0,0,0)'),
                  showlegend=False
             )
        )
        reg_lines.append(
             go.Scatter(
                  x=df['Date'],
                  y=df[upper_min_col],
                  line=dict(color='rgba(0,0,0,0)'),
                  fill='tonexty',
                  fillcolor='rgba(255,0,0,0.1)',
                  showlegend=False
             )
        )

        lower_columns = [col for col in df.columns if col.startswith('Lower')]
        lower_max_col = max(lower_columns, key=lambda x: df[x].max())
        lower_min_col = min(lower_columns, key=lambda x: df[x].min())

        reg_lines.append(
             go.Scatter(
                  x=df['Date'],
                  y=df[lower_min_col],
                  line=dict(color='rgba(0,0,0,0)'),
                  showlegend=False
             )
        )
        reg_lines.append(
             go.Scatter(
                  x=df['Date'],
                  y=df[lower_max_col],
                  line=dict(color='rgba(0,0,0,0)'),
                  fill='tonexty',
                  fillcolor='rgba(0,255,0,0.1)',
                  showlegend=False
             )
        )

        buy_signals = signal_df[signal_df['Signal'] == 'Buy']
        sell_signals = signal_df[signal_df['Signal'] == 'Sell']

        reg_lines.append(
             go.Scatter(
                  x=buy_signals['Date'],
                  y=buy_signals['Close'],
                  mode='markers',
                  marker=dict(symbol='triangle-up', color='green', size=10),
                  name='BUY'
             )
        )
        reg_lines.append(
             go.Scatter(
                  x=sell_signals['Date'],
                  y=sell_signals['Close'],
                  mode='markers',
                  marker=dict(symbol='triangle-down', color='red', size=10),
                  name='SELL'
             )
        )

        frames.append(go.Frame(data=[
            go.Candlestick(
                x=df['Date'].iloc[:i],
                open=df['Open'][:i],
                high=df['High'][:i],
                low=df['Low'][:i],
                close=df['Close'][:i]
            ),
            *reg_lines
        ], name=str(i)))

    fig = go.Figure(
        data=[initial_plot] + reg_lines,
        layout=go.Layout(
            xaxis=dict(title='', rangeslider=dict(visible=False)),
            title='Repainting in Regression Strategies',
            width=900,
            height=650,
            updatemenus=[
            dict(
                type='buttons',
                buttons=[play_button, pause_button],
                direction='left',
                x=0.5,
                y=1.15,
                xanchor='center',
                yanchor='top',
            )
            ]
        ),
        frames=frames
    )

    fig.show()

ticker='BTC-USD'
start_date='2018-01-01'
interval='1wk'

df = yf.download(
    tickers=ticker,
    start=start_date,
    interval=interval,
    progress=False
)

regression_repainting(
     df,
     logarithmic=True,
     initial_sample_size=20,
     significance_levels=[0.01, 0.05]
)