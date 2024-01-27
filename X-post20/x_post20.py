import os
import shutil
from PIL import Image
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mplfinance as mpf

def generate_images(df, target_folder, non_target_folder, last_n_bar_folder, bar_length=10):

    df = df.drop(df.tail(1).index)

    for i in range(len(df) - bar_length + 1):
        ohlc_data = df.iloc[i:i+bar_length]

        data = ohlc_data.iloc[bar_length-3:bar_length]
        open1, open2, open3 = data['Open'].values
        high1, high2, high3 = data['High'].values
        low1, low2, low3 = data['Low'].values
        close1, close2, close3 = data['Close'].values

        is_my_pattern = (
            close1 > open1 and
            close2 > open2 and
            close3 > open3 and
            close1 < close2 and
            close2 < close3 and
            high1 < high2 and
            high2 < high3 and
            low1 < low2 and
            low2 < low3 and
            ((high1 - close1) + (open1 - low1)) < (close1 - open1) and
            ((high2 - close2) + (open2 - low2)) < (close2 - open2) and
            ((high3 - close3) + (open3 - low3)) < (close3 - open3) and
            close3 > max(ohlc_data.iloc[:bar_length-1]['High'])
        )

        folder_name = target_folder if is_my_pattern else non_target_folder

        if i == len(df) - bar_length:
            folder_name = last_n_bar_folder

        mpf.plot(
            ohlc_data,
            type='candle',
            style='yahoo',
            axisoff=True,
            savefig=dict(
                fname = f'./{folder_name}/{ticker}{"_ohlc_target_" if is_my_pattern else "_ohlc_"}{i+1}.png',
                bbox_inches='tight'
            )
        )

last_n_bar_folder_name='classification_imgs/last_n_bar_imgs'
pre_target_folder_name='classification_imgs/pre_target_imgs'
target_folder_name='classification_imgs/target_imgs'
non_target_folder_name='classification_imgs/non_target_imgs'
bar_length=10
tickers = [
    'BTC-USD',
    'ETH-USD',
    'DOGE-USD',
    'BNB-USD',
    'SOL-USD',
    'XRP-USD',
    'INJ-USD',
    'LINK-USD',
    'ADA-USD',
    'AVAX-USD'
]
start_date = (datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d')
interval = '1d'

for ticker in tickers:
    df = yf.download(
        tickers=ticker,
        start=start_date,
        interval=interval,
        progress=False
    )[['Open', 'High', 'Low', 'Close']]
    generate_images(
        df,
        target_folder_name,
        non_target_folder_name,
        last_n_bar_folder_name,
        bar_length
    )

def organize_images(tickers, target_folder, non_target_folder, pre_target_folder):
    for filename in os.listdir(target_folder):
        for ticker in tickers:
            parts = filename.split('_')
            if ticker == parts[0]:
                index = int(parts[-1].split('.')[0])
                prev_filename = f'{ticker}_ohlc_{index - 3}.png'
                new_target_filename = f'{ticker}_ohlc_{index}.png'
                new_target_path = os.path.join(target_folder, new_target_filename)

                if not os.path.exists(new_target_path):
                    if os.path.exists(os.path.join(non_target_folder, prev_filename)):
                        os.rename(
                            os.path.join(non_target_folder, prev_filename),
                            os.path.join(non_target_folder, f'{ticker}_ohlc_target_{index - 3}.png')
                        )
                    os.rename(
                        os.path.join(target_folder, filename),
                        new_target_path
                    )

    for filename in os.listdir(non_target_folder):
        if "target" in filename:
            source_path = os.path.join(non_target_folder, filename)
            destination_path = os.path.join(pre_target_folder, filename)
            shutil.move(source_path, destination_path)

organize_images(tickers, target_folder_name, non_target_folder_name, pre_target_folder_name)

def select_and_remove_files(pre_target_folder, non_target_folder):
    pre_target_count = len(os.listdir(pre_target_folder))
    non_target_files = os.listdir(non_target_folder)

    selected_non_target_files = random.sample(non_target_files, pre_target_count)

    for file_name in non_target_files:
        if file_name not in selected_non_target_files:
            file_path = os.path.join(non_target_folder, file_name)
            os.remove(file_path)

select_and_remove_files(pre_target_folder_name, non_target_folder_name)

def resize_images(folder, width=128, height=128):
    for fn in os.listdir(folder):
        if fn.endswith('.png'):
            image_path = os.path.join(folder, fn)
            image = Image.open(image_path)
            resized_image = image.resize((width, height))
            resized_image_path = os.path.join(folder, fn)
            resized_image.save(resized_image_path)

folders = [pre_target_folder_name, non_target_folder_name, last_n_bar_folder_name]
for folder in folders:
    resize_images(folder)

def predict_target_images(non_target_folder, pre_target_folder, last_n_bar_folder):
    non_target_features = os.listdir(non_target_folder)
    pre_target_features = os.listdir(pre_target_folder)

    non_target_features = [file for file in non_target_features if file.endswith('.png')]
    pre_target_features = [file for file in pre_target_features if file.endswith('.png')]

    features = []
    for path in [os.path.join(non_target_folder, file) for file in non_target_features]:
        img = Image.open(path)
        img_array = np.array(img).flatten()
        features.append(img_array)

    for path in [os.path.join(pre_target_folder, file) for file in pre_target_features]:
        img = Image.open(path)
        img_array = np.array(img).flatten()
        features.append(img_array)

    X = np.array(features)
    y = np.array([1] * len(pre_target_features) + [0] * len(non_target_features))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: {:.1f}%".format(accuracy * 100))

    new_images = os.listdir(last_n_bar_folder)
    new_image_paths = [os.path.join(last_n_bar_folder, img) for img in new_images]

    print("***PREDICTIONS***")

    for image_path in new_image_paths:
        ticker = image_path.split('\\')[-1].split('_')[0]

        img = Image.open(image_path)
        img_array = np.array(img).flatten()
        img_array = img_array.reshape(1, -1)

        prediction = model.predict(img_array)

        if prediction == 1:
            print(f"{ticker}: Target")
        else:
            print(f"{ticker}: Non-Target")

predict_target_images(non_target_folder_name, pre_target_folder_name, last_n_bar_folder_name)