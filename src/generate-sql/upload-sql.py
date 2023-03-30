"""upload csv data to postgres
upload-sql.py
email: blum.da@northeastern.edu
"""

import os
import pandas as pd
import psycopg2
import yaml

from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL


def main():
    load_dotenv(Path.cwd() / ".env")
    server_creds = {
        "drivername": "postgresql+psycopg2",
        "host": os.getenv("PG_HOST"),
        "port": os.getenv("PG_PORT"),
        "database": os.getenv("PG_DATABASE"),
        "username": os.getenv("PG_ADMIN"),
        "password": os.getenv("PG_ADMIN_PASSWORD"),
    }
    yaml_file = Path.cwd() / "res" / "flights_table.yml"
    with open(yaml_file, "r") as f:
        labels = yaml.safe_load(f)
    # test_csv = Path.cwd() / "res" / "test.csv"
    flight_2018_raw_csv = Path.cwd() / "raw" / "Flights_2018_1.csv"
    upload_file(
        credentials=server_creds,
        table_name=labels["table_name"],
        # csv_file=test_csv,
        csv_file=flight_2018_raw_csv,
        labels=labels["columns"],
    )


def upload_file(
    credentials: dict[str, str],
    table_name: str,
    csv_file: Path,
    labels: dict[str, str],
):
    columns = list(labels.keys())
    df = pd.read_csv(csv_file, low_memory=False)
    reduced_df = df[columns]
    server_url = URL.create(**credentials)
    try:
        db = create_engine(server_url)
        conn = db.connect()
        reduced_df.to_sql(table_name, con=conn, if_exists="replace")
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


if __name__ == "__main__":
    main()
