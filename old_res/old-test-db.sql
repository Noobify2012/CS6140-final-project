-- CREATE DATABASE flights;
DROP TABLE IF EXISTS demo;
CREATE TABLE demo (
    id SERIAL PRIMARY KEY,
    title VARCHAR NOT NULL
);
INSERT INTO demo(id, title) VALUES
    (1, 'testing'),
    (2, 'today');