import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
DATA_DIR = os.path.join(BASE_DIR, 'get_data')

df = pd.read_csv(os.path.join(DATA_DIR, 'matches.csv'))

#checking column types and null values
df.info()
df.isna().sum()

to_int = ['season', 'match_day', 'home_score', 'away_score']

#turning columns into integers
for col in to_int:
    df[col] = df[col].astype(int)

#changing from utc to datetime
df['date'] = pd.to_datetime(pd.to_datetime(df.date, utc = True).dt.strftime("%Y-%m-%d"))

def get_team_points(x, team):

    #home points    
    home_df = df[(df.home_team == team) & (df.date < x.date) & (df.season == x.season)]
    h_points = home_df.winner.value_counts().reset_index()
    h_points.columns = ['type', 'winner']
    h_points['point'] = np.where(h_points['type'] == 'HOME_TEAM', h_points.winner * 3, np.where(h_points['type'] == 'DRAW', h_points.winner, 0))
    home_points = h_points.point.sum()

    #away points
    away_df = df[(df.away_team == team) & (df.date < x.date) & (df.season == x.season)]
    a_points = away_df.winner.value_counts().reset_index()
    a_points.columns = ['type', 'winner']
    a_points['point'] = np.where(a_points['type'] == 'AWAY_TEAM', a_points.winner * 3, np.where(a_points['type'] == 'DRAW', a_points.winner, 0))
    away_points = a_points.point.sum()

    total_points = home_points + away_points

    #home points in last 3 games
    h_points = home_df[(home_df.match_day >= (x.match_day - 3)) & (home_df.match_day < x.match_day)].winner.value_counts().reset_index()
    h_points.columns = ['type', 'winner']
    h_points['point'] = np.where(h_points['type'] == 'HOME_TEAM', h_points.winner * 3, np.where(h_points['type'] == 'DRAW', h_points.winner, 0))
    home_l_points = h_points.point.sum()

    #away points in last 3 games
    a_points = away_df[(away_df.match_day >= (x.match_day - 3)) & (away_df.match_day < x.match_day)].winner.value_counts().reset_index()
    a_points.columns = ['type', 'winner']
    a_points['point'] = np.where(a_points['type'] == 'HOME_TEAM', a_points.winner * 3, np.where(a_points['type'] == 'DRAW', a_points.winner , 0))
    away_l_points = a_points.point.sum()

    total_l_points = home_l_points + away_l_points

    return total_points , home_points , away_points, total_l_points

def get_team_goals(x, team):

    #get team home goals made and suffered
    home_df = df[(df.date < x.date) & (df.home_team == team) & (df.season == x.season)]
    home_goals = home_df['home_score'].sum()
    home_goals_suffered = home_df['away_score'].sum()      

    #get team away goals
    away_df = df[(df.date < x.date) & (df.away_team == team) & (df.season == x.season)]
    away_goals = away_df['away_score'].sum()    
    away_goals_suffered = away_df['home_score'].sum()     

    #total goals made
    total_goals = home_goals + away_goals

    #total goals suffered
    total_goals_suffered = home_goals_suffered + away_goals_suffered

    return total_goals, total_goals_suffered

def get_rank(x):
    home_df = x.groupby(['season', 'match_day', 'home_team']).min()[['ht_total_points', 'ht_goals', 'ht_goals_sf']].reset_index()
    home_df.columns = ['season', 'match_day', 'team', 'total_points', 'goals', 'goals_sf']

    away_df = x.groupby(['season', 'match_day', 'away_team']).min()[['at_total_points', 'at_goals', 'at_goals_sf']].reset_index()
    away_df.columns = ['season', 'match_day', 'team', 'total_points', 'goals', 'goals_sf']

    rank_df = pd.concat([home_df, away_df], ignore_index = True)
    rank_df['goals_df'] = rank_df.goals - rank_df.goals_sf
    rank_df['rank'] = rank_df.sort_values(['total_points', 'goals_sf', 'goals']).total_points.rank(method = 'first', ascending = False)

    home_team_rank = rank_df[rank_df.team == x.home_team].min()['rank']
    away_team_rank = rank_df[rank_df.team == x.away_team].min()['rank']

    return home_team_rank, away_team_rank

#points so far and points in latest 3 games
df[['ht_total_points', 'ht_home_points','ht_away_points', 'ht_ls_points']] = pd.DataFrame(
    df.apply(
        lambda x: get_team_points(x, x.home_team), axis = 1).to_list(), index = df.index)

df[['at_total_points', 'at_home_points','at_away_points', 'at_ls_points']] = pd.DataFrame(
    df.apply(
        lambda x: get_team_points(x, x.away_team), axis = 1).to_list(), index = df.index)
     

#goals so far made and suffered
df[['ht_goals', 'ht_goals_sf']] = pd.DataFrame(
    df.apply(
        lambda x: get_team_goals(x, x.home_team), axis = 1).to_list(), index = df.index)

df[['at_goals', 'at_goals_sf']] = pd.DataFrame(
    df.apply(
        lambda x: get_team_goals(x, x.away_team), axis = 1).to_list(), index = df.index)


#get position in table
df[['ht_ranking', 'at_ranking']] = pd.DataFrame(
    df.apply(
        lambda x: get_rank(x), axis = 1).to_list(), index = df.index)

df.head()
#goals in latest games -------------------


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

