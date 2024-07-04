import os
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

PATH = "/TimeseriesDatasets/forecasting/fred/"
OUTPUT_PATH = PATH + "preprocessed/"
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)


print("Loading FREDInfo.csv and ts_meta.csv")
df_info = pd.read_csv(PATH + "FREDInfo.csv")
df_meta = pd.read_csv(PATH + "ts_meta.csv")

print("Loading fred-complete.npz")
data_complete = np.load(PATH + "fred-complete.npz", allow_pickle=True)

print("Mergning data")
df_final = df_info
seasonalities = {s: dict() for s in df_final["SP"].unique()}

for i, ((index, info), series_complete) in tqdm(
    enumerate(zip(df_final.iterrows(), data_complete)), total=len(df_final)
):
    # get series ID and extract its metadata
    fred_id = info["FREDid"]
    metadata = df_meta[df_meta["id"] == fred_id]
    df_final.loc[index, metadata.columns] = metadata.iloc[0]

    # get seasonality and save series
    seasonality = df_final.at[df_final[df_final["FREDid"] == fred_id].index[0], "SP"]
    series_complete_clean = series_complete[~np.isnan(series_complete)]  # remove nans
    seasonalities[seasonality][fred_id] = series_complete_clean

df_final.drop(["id"], axis=1, inplace=True)

print("Saving df_final")
df_final.to_csv(OUTPUT_PATH + "FRED_meta.csv", index=False)

print("Saving series")
for seasonality, series in seasonalities.items():
    print(f"{seasonality}: {len(series)}")
    np.save(OUTPUT_PATH + f"FRED_{seasonality}.npy", series)
