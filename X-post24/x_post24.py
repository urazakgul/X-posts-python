import pandas as pd
from requests_html import HTMLSession

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

tickers = fetch_crypto_data(num_pages=3)