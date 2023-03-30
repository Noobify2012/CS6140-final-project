import psycopg2

server_creds = {
    "host": "localhost",
    "database": "flights_db",
    "user": "postgres",
    "password": "postgres",
}


try:
    conn = psycopg2.connect(**server_creds)
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM pg_catalog.pg_tables;")
    print(cur.fetchone())
    # cur.execute(f"SELECT * FROM pg_catalog.pg_tables WHERE schemaname='flights';")
    cur.execute(
        f"SELECT column_name FROM information_schema.columns WHERE table_name = 'flights';"
    )
    column_names = cur.fetchall()
    print(column_names)

except (Exception, psycopg2.Databaseerroror) as error:
    print(error)
