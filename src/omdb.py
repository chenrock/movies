import urllib.request, urllib.parse, urllib.error
import json


with open('api_key.json') as f:
    keys = json.load(f)
    omdb_api = keys['omdb_api']

url = 'http://www.omdbapi.com/?'
api_key = '&apikey='+omdb_api