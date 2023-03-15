#!/usr/bin/env python

# generate-sql.py
# email: blum.da@northeastern.edu
# generates an sql DDL for Postgres

import ast
from pathlib import Path

def try_eval(val):
    r = ''
    try:
        r = ast.literal_eval(val)
    except Exception:
        return val 
    return r


def get_postgres_type(csv_field : str) -> str:
    '''Tests the type of a str in python and returns its postgres equivalent
    '''
    stripped = csv_field.replace('"', '').strip()
    t = try_eval(stripped)
    match t:
        case int():
            return 'BIGINT'
        case bool():
            return 'BOOLEAN'
        case float():
            return 'DOUBLE PRECISION'
        case _:
            if stripped != '':
                return 'VARCHAR'
            else:
                return '__FILL ME__'





file_name = 'test.csv'
current_dir = Path(__file__).parent.resolve()
csv_file = current_dir / file_name
print(f'Reading file {csv_file}')

with open(csv_file, 'r') as file:
    headers = file.readline().split(",")[:-1]
    first_line = file.readline().split(",")[:-1]

file_start = '''DROP TABLE IF EXISTS flights;
CREATE TABLE flights (
    id      SERIAL      PRIMARY KEY
'''
file_end = ');'
spacer = '    '

for (head, col) in zip(headers, first_line):
    part_1 = f'{spacer}{head.strip():50} '
    part_2 = f'{get_postgres_type(col):20},\n'
    file_start += part_1 + part_2

sql_file = current_dir / 'create-flights.sql'
with open(sql_file, 'w') as file:
    file.writelines(file_start + file_end)







