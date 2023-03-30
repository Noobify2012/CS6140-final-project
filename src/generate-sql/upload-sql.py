"""upload csv data to postgres
upload-sql.py
email: blum.da@northeastern.edu
"""

import os
import pandas as pd
import psycopg2

from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL



def main():
    load_dotenv(Path.cwd() / '.env')
    server_creds = {
        "drivername": 'postgresql+psycopg2',
        "host": os.getenv("PG_HOST"),
        "port": os.getenv("PG_PORT"),
        "database": os.getenv("PG_DATABASE"), "username": os.getenv("PG_ADMIN"),
        "password": os.getenv("PG_ADMIN_PASSWORD")
    }
    labels = {
        "Year": "SMALLINT" ,
        "Month": "SMALLINT",
        "DayofMonth": "SMALLINT"
    }
    columns = list(labels.keys())

    test_csv = Path.cwd() / "res" / "test.csv"
    df = pd.read_csv(test_csv, low_memory=False)
    reduced_df = df[columns]
    print(reduced_df.columns.values)

    server_url = URL.create(**server_creds)
    print(server_url)

    try:
        db = create_engine(server_url)
        conn = db.connect()
        # conn = psycopg2.connect(**server_creds)
        reduced_df.to_sql('flights', con=conn, if_exists='replace')
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


if __name__ == '__main__':
    main()