import pandas as pd
import requests

#prediction request url
#url = 'https://rental-properties.herokuapp.com/predict'
url = 'http://192.168.0.9:5000/predict'

#headers
headers = {'Content-type': 'application/json'}

#input example
input_df = pd.DataFrame({'h_odd': [1.68], 'd_odd' : [4.22], 'a_odd' : [4.96], 'ht_rank': [8],
                  'ht_l_points' : [3], 'at_rank' : [16] , 'at_l_points' : [1.33], 'at_l_wavg_points' : [1.96],
                  'at_l_wavg_goals' : [2.01], 'at_l_wavg_goals_sf': [1.54], 'at_win_streak': [1], 'ls_winner' : ['AWAY_TEAM']})

#transform df into json format
df_json = input_df.to_json(orient = 'records')

#make request to server
r = requests.post( url = url , data = df_json, headers =headers)

#output dataframe with prediction
output = pd.DataFrame(r.json(), columns = r.json()[0].keys())

#Print result
print(output.prediction[0])