import pywhatkit as kit
import yfinance as yf
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.style.use('fivethirtyeight')

ticker = 'BTC-USD'
start_date = datetime.now().strftime('%Y-%m-%d')
interval = '1m'
base_column = 'Adj Close'

data = yf.download(
    tickers=ticker,
    start=start_date,
    interval=interval,
    progress=False
)

data.index = data.index.tz_localize(None)
data['Return'] = data[f'{base_column}'].pct_change()

excel_file_path = f'{ticker}_data_{datetime.now().strftime("%Y%m%d")}.xlsx'
data.to_excel(excel_file_path)

plt.figure(figsize=(12, 16))

plt.subplot(2, 1, 1)
plt.plot(data.index, data[f'{base_column}'], color='red')
plt.title(f'{ticker} - {base_column} for {interval} Intervals ({datetime.now().strftime('%d/%m/%Y %H:%M:%S')})')
plt.xlabel('')
date_format = mdates.DateFormatter('%H:%M')
plt.gca().xaxis.set_major_formatter(date_format)
plt.grid(True)

plt.subplot(2, 1, 2)
plt.hist(data['Return'].dropna(), bins=50, color='skyblue', edgecolor='black')
plt.title(f'Distribution of Returns for {ticker}')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.grid(True)

plt.tight_layout()
plt.savefig('combined_plot.jpg')
plt.close()

to_whatsapp_number = 'Your_WhatsApp_Number'
message = 'Yo, the data has been updated.\n' + datetime.now().strftime('%d/%m/%Y %H:%M:%S')
plot_path = 'combined_plot.jpg'
kit.sendwhats_image(
    receiver=to_whatsapp_number,
    img_path=plot_path,
    caption=message,
    tab_close=True,
    close_time=60
)