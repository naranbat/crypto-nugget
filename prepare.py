import pandas as pd
import glob


symbol = "BTCUSDT"
interval = "1d"
market = "spot"
dir_name = f"./data/spot/{symbol}"


files = glob.glob(f"./data/{market}/{symbol}/{symbol}-{interval}*.zip", recursive=True)
files = sorted(files)

columns = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
]

dfs = [] # List to hold individual dataframes

# Loop through each file in the specified directory
for file_path in files:
    df = pd.read_csv(file_path, compression='zip', names=columns)
    dfs.append(df)

# Concatenate all the individual DataFrames into one large DataFrame
if dfs:
    full_dataset = pd.concat(dfs, ignore_index=True)
    full_dataset["open_time"] = pd.to_datetime(full_dataset["open_time"], unit='ms')
    full_dataset["close_time"] = pd.to_datetime(full_dataset["close_time"], unit='ms')

    full_dataset = full_dataset[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    full_dataset = full_dataset.sort_values("open_time")
    full_dataset = full_dataset.set_index("open_time")

    print("Successfully loaded all data into a single DataFrame.")
    print(full_dataset.head()) # Display the first few rows
    print(f"Total rows: {len(full_dataset)}")

    full_dataset.to_parquet("./data/cache/normalized.parquet")
else:
    print("No CSV files found in the zip archives.")