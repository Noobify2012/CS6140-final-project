"""create DDL script for postgres from csv files
create-ddl.py
email: blum.da@northeastern.edu
"""
import pandas as pd

from pathlib import Path
from textwrap import dedent


def main(file):
    # df = pd.read_csv(file, low_memory=False)
    # df = df[headers] if headers else df
    # print(df.iloc[:2])
    # labels = list(df.columns.values)
    # types = list(df.dtypes)
    # print(types)
    labels = {
        "Year": "SMALLINT" ,
        "Month": "SMALLINT",
        "DayofMonth": "SMALLINT"
    }
    extras = {
        "Year": "NOT NULL"
    }

    ddl_text = generate_ddl_text(labels=labels, extras=extras)
    file_name = Path.cwd() / 'res' / 'create-db.sql'
    output_ddl_file(ddl_text=ddl_text, file_path=file_name, overwrite=True)



def generate_ddl_text(labels, create_id=True, extras={}, ending=''):
    spacer = '    '
    file_start = f"DROP TABLE IF EXISTS flights;\nCREATE TABLE flights (\n"
    file_end = ');'
    insert_string = spacer + "{label:50}{type:20}{extra}\n"
    ddl_string = file_start

    if create_id:
        ddl_string += insert_string.format(label="id", type="SERIAL", extra="PRIMARY KEY")

    extra_keys = extras.keys()
    for label, type in labels.items():
        extra = extras[label] if label in extra_keys else '' 
        ddl_string += insert_string.format(label=label, type=type, extra=extra)
    
    ddl_string += ending + file_end
    return ddl_string

def output_ddl_file(ddl_text, file_path, overwrite=False):
    write_option = 'w' if overwrite else 'x'
    with open(file_path, write_option) as f:
        f.write(ddl_text)

        






if __name__ == '__main__':
    file = Path.cwd() / 'res' / 'test.csv'
    main(file)
    # generate_ddl(file)
