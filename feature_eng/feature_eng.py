import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
DATA_DIR = os.path.join(BASE_DIR, 'get_data')

df = pd.read_csv(os.path.join(DATA_DIR, 'matches.csv'))

to_int = ['season', 'match_day', 'home_score', 'away_score']

#turning columns into integers
for col in to_int:
    df[col] = df[col].astype(int)

#changing from utc to datetime
df['date'] = pd.to_datetime(pd.to_datetime(df.date, utc = True).dt.strftime("%Y-%m-%d"))


def get_team_points(x, team):

    #home points    
    h_points = df[(df.home_team == team) & (df.date < x.date) & (df.season == x.season)].winner.value_counts().reset_index()
    h_points.columns = ['type', 'winner']
    h_points['point'] = np.where(h_points['type'] == 'HOME_TEAM', h_points.winner * 3, np.where(h_points['type'] == 'DRAW', 1, 0))
    home_points = h_points.point.sum()

    #away points
    a_points = df[(df.away_team == team) & (df.date < x.date) & (df.season == x.season)].winner.value_counts().reset_index()
    a_points.columns = ['type', 'winner']
    a_points['point'] = np.where(a_points['type'] == 'AWAY_TEAM', a_points.winner * 3, np.where(a_points['type'] == 'DRAW', 1, 0))
    away_points = a_points.point.sum()

    total_points = home_points + away_points

    return total_points , home_points , away_points

#points so far
df[['home_t_total_points', 'home_t_home_points','home_t_away_points']] = pd.DataFrame(
    df.apply(
        lambda x: get_team_points(x, x.home_team), axis = 1).to_list(), index = df.index)

df[['home_t_total_points', 'home_t_home_points','home_t_away_points']] = pd.DataFrame(
    df.apply(
        lambda x: get_team_points(x, x.home_team), axis = 1).to_list(), index = df.index)        



#points in latest games
#goals so far
#goals in latest games
#position in the competition
#amount of victories
#amount of draws
#amount of losses
#amount of home w
#amount of away w
#amount of home d
#amount of away d
#amount of home l
#amount of away l
#history between two teams
#last season's position
#last season's position at current round
#streak of wins
#streak of draws
#streak of losses
#days between last game
#result between last game of the teams
#goals pro
#goals against
#goals balance
#maybe create my own metric to give weight to the teams

