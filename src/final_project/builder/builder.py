"""builder.py
message pandas dataframes the way we like them to get messaged.
"""

import pandas as pd
import numpy as np
# from sklearn.preprocessing import OneHotEncoder


def cyclical_encode_dmy(df: pd.DataFrame) -> pd.DataFrame:
    # sin_value = np.sin(2*np.pi*feature_value/feature_range)
    # cos_value = np.cos(2*np.pi*feature_value/feature_range)
    #Year, Month, DayofMonth,
    # if df['Month']== 1 | df['Month']== 3 | df['Month']== 5 | df['Month']== 7 | df['Month']== 8 | df['Month']== 10 | df['Month']== 12 :
    #     df['date_sin'] = np.sin(2*np.pi*df['DayofMonth']/31)
    #     df['date_cos'] = np.cos(2*np.pi*df['DayofMonth']/31)
    # elif df['Month']== 4 | df['Month']== 6 | df['Month']== 9 | df['Month']== 11:
    #     df['date_sin'] = np.sin(2*np.pi*df['DayofMonth']/30)
    #     df['date_cos'] = np.cos(2*np.pi*df['DayofMonth']/30)
    # elif df['Month']== 2 & df['Year']== 2020: 
    #     df['date_sin'] = np.sin(2*np.pi*df['DayofMonth']/29)
    #     df['date_cos'] = np.cos(2*np.pi*df['DayofMonth']/29)
    # else: 
    #     df['date_sin'] = np.sin(2*np.pi*df['DayofMonth']/28)
    #     df['date_cos'] = np.cos(2*np.pi*df['DayofMonth']/28)
    df["date_sin"] = np.sin(2 * np.pi * df["DayofMonth"] / 31)
    df["date_cos"] = np.cos(2 * np.pi * df["DayofMonth"] / 31)
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["year_sin"] = np.sin(2 * np.pi * df["Year"] / 100)
    df["year_cos"] = np.cos(2 * np.pi * df["Year"] / 100)
    return df.drop(["DayofMonth", "Month", "Year"], axis=1)


def one_hot_encoding(
    df: pd.DataFrame, column_name: str, prefix_name: str
) -> pd.DataFrame:
    # enc = OneHotEncoder(handle_unknown='ignore').set_output(transform="pandas")
    # enc.fit(df)
    # print(enc.categories_)
    # encoded_frame = enc.transform(df)
    # print(encoded_frame)
    # encoded_frame = set_output
    return pd.get_dummies(df, columns=column_name, prefix=prefix_name)
