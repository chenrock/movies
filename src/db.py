import sqlite3
import pandas as pd
import json
import requests
import pickle


def db_tables_to_csv(db_file, output_folder):
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
    return 'The db_tables_to_csv function completed running. Check output folder for CSVs of database tables.'


# ----- create csv for single table in original movies.db database
db_tables_to_csv('data/movies.db', 'data/')


# ----- read in movies csv table as df
movies_df = pd.read_csv('data/db-csv-table_movies.csv')
# ----- dropping 'index' and 'id' columns leaving 33 dimensions as noted in 'CEE Data Scientist Case Study 2019_06.pdf'
movies_df = movies_df.drop(columns=['index', 'id'])


# ----- extracting IMDb ID from 'movie_imdb_link' column in movies_df
movie_imdb_link_list = movies_df['movie_imdb_link'].to_list()
imdb_id_list = []
for link in movie_imdb_link_list:
    imdb_id_list.append(link.split('/')[-2])


# ----- get OMDb API key
with open('api_key.json') as f:
    keys = json.load(f)
    omdb_api = keys['omdb_api']

# ----- set up url pieces of omdb data extract
url = 'http://www.omdbapi.com/?i='
api_key = '&apikey='+omdb_api

# ----- set up progress tracker for omdb data extract
x25 = int(len(imdb_id_list)/4)
x50 = int(x25*2)
x75 = int(x25*3)

# ----- extract data from omdb
omdb_data = []
for count, imdb_id in enumerate(imdb_id_list, 1):
    try:
        page = requests.get(url + imdb_id + api_key)
        omdb_data.append(page.content)
    except:
        omdb_data.append("")
    if count == x25:
        print('25% complete')
    elif count == x50:
        print('50% complete')
    elif count == x75:
        print('75% complete')
    elif count == len(imdb_id_list):
        print('100% complete')


# ----- check to see if the extract was able to grab json like string for each movie
[i for i,x in enumerate(omdb_data) if not x.startswith(b'{')]
# ----- investigate because the check above found one movie (index 4525) without json like string
print(omdb_data[4525])
page = requests.get(url + imdb_id_list[4525] + api_key)
print(page.content)
# ----- was able to print content above so replacing the 4525 index in list with omdb json data
omdb_data[4525] = page.content
print(omdb_data[4525])
[i for i,x in enumerate(omdb_data) if not x.startswith(b'{')]

# ----- create pickle file of omdb extract
with open('data/omdb_extract.pkl', 'wb') as f:
    pickle.dump(omdb_data, f)

# ----- load omdb extract pickle
with open('data/omdb_extract.pkl', 'rb') as f:
    omdb_data = pickle.load(f)

# ----- converting to list of json objects
omdb_data_json = []
for idx, d in enumerate(omdb_data):
    try:
        omdb_data_json.append(json.loads(d))
    except:
        omdb_data_json.append(json.loads(b'{}'))
        print("index", idx, "appended b'{}' to omdb_data_json list")

# ----- additional data fields to extract from omdb
list_keys=['Title', 'imdbID', 'Writer', 'Plot', 'Awards', 'Metascore', 'Type']

# ----- create list of extra omdb data to help create df
extra_omdb_data_list = []
for m in omdb_data_json:
    extra_omdb_data_sublist = []
    for k in list_keys:
        try:
            extra_omdb_data_sublist.extend([m[k]])
        except:
            extra_omdb_data_sublist.extend("")
    extra_omdb_data_list.append(extra_omdb_data_sublist)

# ----- create df from extra omdb data
extra_omdb_data_df = pd.DataFrame.from_records(extra_omdb_data_list, columns=list_keys)


# ----- upload df to db as table
# ----- create database connection
db = sqlite3.connect('data/movies.db')
# ----- write records stored in df to db
extra_omdb_data_df.to_sql('extra_omdb_data', db, if_exists='replace', index=False)

# ----- create csv for all tables in original movies.db database
db_tables_to_csv('data/movies.db', 'data/')
