import pandas as pd

parquet_file = 'res/Combined_Flights_2018.parquet'
pd.read_parquet(parquet_file, engine='pyarrow')

