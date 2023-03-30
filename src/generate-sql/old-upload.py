"""upload csv files to a postgres database
generate-sql.py
email: blum.da@northeastern.edu
"""

import ast
import dateutils
import pandas as pd
from pathlib import Path


def try_eval(val):
    r = ""
    try:
        r = ast.literal_eval(val)
    except Exception:
        return val
    return r


def get_postgres_type(csv_field: str) -> str:
    """Tests the type of a str in python and returns its postgres equivalent"""
    stripped = csv_field.replace('"', "").strip()
    t = try_eval(stripped)
    match t:
        case int():
            return "BIGINT"
        case bool():
            return "BOOLEAN"
        case float():
            return "DOUBLE PRECISION"
        case _:
            if stripped != "":
                return "VARCHAR"
            else:
                return "__FILL ME__"


# SCRIPT ---------------------------------------------------------------------

# file_name = 'test.csv'
# res_folder = 'res'
# current_dir = Path(__file__).parent.resolve()
# res_dir = current_dir.parent.parent.resolve() / res_folder
# csv_file = res_dir / file_name
# print(f'Reading file {csv_file}')

# with open(csv_file, 'r') as file:
#     headers = file.readline().split(",")[:-1]
#     first_line = file.readline().split(",")[:-1]

# file_start = '''DROP TABLE IF EXISTS flights;
# CREATE TABLE flights (
#     id      SERIAL      PRIMARY KEY
# '''
# file_end = ');'
# spacer = '    '

# for (head, col) in zip(headers, first_line):
#     part_1 = f'{spacer}{head.strip():50} '
#     part_2 = f'{get_postgres_type(col):20},\n'
#     file_start += part_1 + part_2

# sql_file = res_dir / 'create-flights.sql'
# print(f'Writing file {sql_file}')
# with open(sql_file, 'w') as file:
#     file.writelines(file_start + file_end)

# sel = 11
# print(headers[sel])
# print(first_line[sel])
# mixed_column_types = {
#     'Originally_Scheduled_Code_Share_Airline': 'str',
#     'IATA_Code_Originally_Scheduled_Code_Share_Airline': 'str',
#     'Div2Airport': 'str',
#     'Div2TailNum': 'str',
# 'Duplicate': 'bool'
# }
# df = pd.read_csv(csv_file, index_col=False, dtype = mixed_column_types, parse_dates=['FlightDate'], nrows=8000)
# df = pd.read_csv(csv_file, index_col=False, nrows=8000)
# df = df.drop(columns=['Unnamed: 119'])
# df['Duplicate'] = df['Duplicate'].map({'N': False, 'Y': True})

shitty_i = 0
single_row = df.iloc[0].to_numpy()
# print(df.iat[0,1])
for name, dtype in df.dtypes.items():
    # item = first_line[shitty_i].replace('"', '').strip()
    item = single_row[shitty_i]
    # item = df.iat[0, shitty_i].replace('"', '').strip()
    df_item = df.iat[0, shitty_i]
    print(
        f"{shitty_i:.<9}{name:.<50}{df_item:.^20}{item:.^20}{str(dtype):.>20}"
    )
    shitty_i += 1


# print(df.iloc[:,[0]])
# print(df.iloc[8180:8900, [0, 11, 13, 57, 86, 93]])

# non_null_df = df.iloc[:,[11,13,86,93]].notna()
# non_null_df = df.iloc[:,[11]].dropna()
# print(df.iloc[:,[11]].dropna())
# print(df.iloc[:,[13]].dropna())
# print(df.iloc[:,[86]].dropna())
# print(df.iloc[:,[93]].dropna())

# sql = df.to_sql('')
