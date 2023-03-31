CREATE DATABASE flights_db;
CREATE USER admin WITH PASSWORD 'admin';
ALTER USER admin WITH SUPERUSER;
\c flights_db;
CREATE TABLE flights(
    id                                                SERIAL              PRIMARY KEY,
    Year                                              SMALLINT            NOT NULL,
    Month                                             SMALLINT            ,
    DayofMonth                                        SMALLINT            
);
CREATE USER read_user WITH PASSWORD 'read_user';
GRANT SELECT ON ALL TABLES IN SCHEMA public TO read_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO read_user;
