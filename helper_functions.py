import pandas as pd
import numpy as np

# Function for reading bitcoin data and creating a dataframe with preprocessed data
def get_bitcoin_df():
    # Read the data
    btc_2017_df = pd.read_csv("data/BTC-2017min.csv")
    btc_2018_df = pd.read_csv("data/BTC-2018min.csv")
    btc_2019_df = pd.read_csv("data/BTC-2019min.csv")
    btc_2020_df = pd.read_csv("data/BTC-2020min.csv")
    btc_2021_df = pd.read_csv("data/BTC-2021min.csv")
    # Concatendate the data while reversing the dataframes (Originally in dataframe 31-12 -> 01-01, we want 01-01 -> 31-12)
    btc_df = pd.concat([btc_2017_df.iloc[::-1], btc_2018_df.iloc[::-1], btc_2019_df[::-1], btc_2020_df[::-1], btc_2021_df[::-1]], axis=0)
    # Remove unneded columns (symbol not needed)
    btc_df = btc_df.drop(['unix', 'symbol'], axis=1)
    # Simplify names
    btc_df.columns = ['date', 'open', 'high', 'low', 'close', 'volume_btc', 'volume_usd']
    # Fill in any missing minutes by re-indexing
    btc_df.index = pd.DatetimeIndex(btc_df['date'])
    all_minutes = pd.date_range(btc_df.index.min(), btc_df.index.max(), freq="T")
    btc_df = btc_df.reindex(all_minutes)
    # Fill in the missing values for the missing minutes
    btc_df['date'] = btc_df.index
    btc_df['open'] = btc_df['open'].interpolate()
    btc_df['high'] = btc_df['high'].interpolate()
    btc_df['low'] = btc_df['low'].interpolate()
    btc_df['close'] = btc_df['close'].interpolate()
    btc_df['volume_btc'] = btc_df['volume_btc'].interpolate()
    btc_df['volume_usd'] = btc_df['volume_usd'].interpolate()
    # Smoothing data / Resampling -> Downsampling (Minutes to Days)
    btc_df_downsample = btc_df[['date', 'open', 'high', 'low', 'close', 'volume_btc', 'volume_usd']].resample('D', on='date').mean().reset_index(drop=False)
    df = btc_df_downsample.copy()
    # Return the created df
    return df

# Function for creating a dataset with look back
def create_dataset_with_look_back(dataset, look_back=1):
    X, Y = [], []
    for i in range(look_back, len(dataset)):
        a = dataset[i-look_back:i, 0]
        X.append(a)
        Y.append(dataset[i, 0])
    return np.array(X), np.array(Y)