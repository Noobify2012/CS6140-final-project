"""create DDL script for postgres from csv files
create-ddl.py
email: blum.da@northeastern.edu
"""
import os
import pandas as pd
import yaml

from dotenv import load_dotenv
from pathlib import Path
from textwrap import dedent


def main():
    load_dotenv(Path.cwd() / ".env")
    yaml_file = Path.cwd() / "res" / "flights_table.yml"
    with open(yaml_file, "r") as f:
        labels = yaml.safe_load(f)
    ddl_text = generate_ddl_text(
        db_name=os.getenv("PG_DATABASE"),
        table_name=labels["table_name"],
        columns=labels["columns"],
        extras=labels["extras"],
    )
    file_name = Path.cwd() / "res" / "create-db.sql"
    output_ddl_file(ddl_text=ddl_text, file_path=file_name, overwrite=True)


def generate_ddl_text(
    db_name, table_name, columns, create_id=True, extras={}, ending=""
):
    # db_name = "flights_db"
    # table_name = "flights"
    spacer = "    "
    file_start = dedent(
        f"""\
    CREATE DATABASE {db_name};
    CREATE USER admin WITH PASSWORD 'admin';
    ALTER USER admin WITH SUPERUSER;
    \c {db_name};
    CREATE TABLE {table_name}(
    """
    )
    file_end = dedent(
        f"""\
    CREATE USER read_user WITH PASSWORD 'read_user';
    GRANT SELECT ON ALL TABLES IN SCHEMA public TO read_user;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO read_user;
    """
    )
    insert_string = spacer + "{label:50}{type:20}{extra},\n"
    ddl_string = file_start

    if create_id:
        ddl_string += insert_string.format(
            label="id", type="SERIAL", extra="PRIMARY KEY"
        )

    extra_keys = extras.keys()
    for column, type in columns.items():
        extra = extras[column] if column in extra_keys else ""
        ddl_string += insert_string.format(
            label=column, type=type, extra=extra
        )

    ddl_string = ddl_string[:-2]
    ddl_string += "\n" + ending + ");\n" + file_end
    return ddl_string


def output_ddl_file(ddl_text, file_path, overwrite=False):
    write_option = "w" if overwrite else "x"
    with open(file_path, write_option) as f:
        f.write(ddl_text)


if __name__ == "__main__":
    # file = Path.cwd() / "res" / "test.csv"
    main()
    # generate_ddl(file)
