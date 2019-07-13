import sqlite3
import pandas as pd


def db_table_to_csv(db_file, output_folder):
    # ----- create database connection
    db = sqlite3.connect(db_file)
    # ----- create cursor using cursor method of the connection object
    cursor = db.cursor()
    # ----- execute select statement select all table names from sqlite_master
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    # ----- call the fetchall() method of the cursor object to fetch the data
    tables = cursor.fetchall()
    # ----- loop through table names to create csv of database table
    for table_name in tables:
        table_name = table_name[0]
        table = pd.read_sql_query("SELECT * from %s" % table_name, db)
        table.to_csv(output_folder + 'db-csv-table_' + table_name + '.csv', index_label='index')
    # ----- close cursor and database connection
    cursor.close()
    db.close()
    return 'The db_table_to_csv function completed running. Check output folder for CSVs of database tables.'

db_table_to_csv('data/movies.db', 'data/')
