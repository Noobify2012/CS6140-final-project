"""builder.py
message pandas dataframes the way we like them to get messaged.
"""

import pandas as pd
import numpy as np


def cyclical_encode_dmy(df: pd.DataFrame) -> pd.DataFrame:
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
    return pd.get_dummies(df, columns=column_name, prefix=prefix_name)
