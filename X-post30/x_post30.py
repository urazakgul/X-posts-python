import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

def plot_event_return(filename, event_range=(-20, 20), target_column='XU100'):
    df = pd.read_excel(filename)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    last_trading_day_indices = df[df['Event'] == 'Last Trading Day'].index

    for i in range(1, abs(event_range[0]) + 1):
        df.loc[last_trading_day_indices - i, 'Event'] = -i

    for i in range(1, event_range[1] + 1):
        df.loc[last_trading_day_indices + i, 'Event'] = i

    df.loc[last_trading_day_indices, 'Event'] = 0

    start_values = df[df['Event'] == event_range[0]][target_column].values
    for i in range(event_range[0], event_range[1] + 1):
        df.loc[df['Event'] == i, f'{target_column}_normalized'] = (df[df['Event'] == i][target_column].values / start_values) * 100

    df['Return'] = df[target_column].pct_change()

    groups = df.groupby(df['Date'].dt.year)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 16))

    for name, group in groups:
        axes[0].plot(group['Event'], group['Return'], color='grey', alpha=0.5)

    avg_line = df.groupby('Event')['Return'].mean()
    axes[0].plot(avg_line.index, avg_line.values, color='red', label='Ortalama', linewidth=2)

    axes[0].set_xlabel('Gün', fontsize=15)
    axes[0].set_ylabel('Getiri', fontsize=15)
    axes[0].set_title(f'{target_column} Endeks Getirileri: 2010-2023 ({event_range[0]},{event_range[1]})', fontsize=20)
    axes[0].legend(loc='best')
    axes[0].set_xticks(range(event_range[0], event_range[1] + 1))
    axes[0].grid(True)
    axes[0].axvline(x=0, color='black', linestyle='--')

    for name, group in groups:
        axes[1].plot(group['Event'], group[f'{target_column}_normalized'], color='gray', alpha=0.5)

    avg_line_normalized = df.groupby('Event')[f'{target_column}_normalized'].mean()
    axes[1].plot(avg_line_normalized.index, avg_line_normalized.values, color='red', label='Ortalama Normalleştirilmiş Değer', linewidth=2)

    axes[1].set_xlabel('Gün', fontsize=15)
    axes[1].set_ylabel('Normalleştirilmiş Değer', fontsize=15)
    axes[1].set_title(f'{target_column} Endeks Normalleştirilmiş Değerleri: 2010-2023 ({event_range[0]},{event_range[1]})', fontsize=20)
    axes[1].legend(loc='best')
    axes[1].set_xticks(range(event_range[0], event_range[1] + 1))
    axes[1].grid(True)
    axes[1].axvline(x=0, color='black', linestyle='--')

    plt.tight_layout()
    plt.show()

plot_event_return('xu100.xlsx', event_range=(-10, 10), target_column='XU100')