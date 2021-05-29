import http.client
import json
import dotenv
import os
#from numpy import set_string_function
import pandas as pd

#loading env variables
dotenv.load_dotenv(dotenv.find_dotenv())

#loading API token
API_TOKEN = os.getenv("API_TOKEN")

#defining all season that have data to make a request
seasons = [2018,2019,2020]

#main url
filter_url = '/v2/competitions/PL/matches?season='

#dataframe that will be populated
df = pd.DataFrame()

#get matches for each season and appends to df
for season in seasons:
    connection = http.client.HTTPConnection('api.football-data.org')
    headers = { 'X-Auth-Token': API_TOKEN }
    connection.request('GET', filter_url + str(season) , None, headers )
    response = json.loads(connection.getresponse().read().decode())

    matches = [i for i in response['matches']]

    for match in matches:
        df = df.append(
        {
        'season' : season,
        'date' : match['utcDate'],
        'status' : match['status'],
        'match_day' : match['matchday'],
        'home_team' : match['homeTeam']['name'],
        'away_team' : match['awayTeam']['name'],
        'home_score' : match['score']['fullTime']['homeTeam'],
        'away_score' : match['score']['fullTime']['awayTeam'],
        'winner' : match['score']['winner']
        }
        , ignore_index = True)

    print(season, 'done!')

cols_order = ['season', 'date', 'status', 'match_day', 'home_team', 'away_team', 'home_score',
                'away_score', 'winner']

df = df[cols_order]

df.to_csv('matches.csv', index = False)
