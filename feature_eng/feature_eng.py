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

#home points made in each match
df['h_match_points'] = np.where(df['winner'] == 'HOME_TEAM', 3 , np.where(df['winner'] == 'DRAW',1, 0))

#away points made in each match
df['a_match_points'] = np.where(df['winner'] == 'AWAY_TEAM', 3 , np.where(df['winner'] == 'DRAW',1, 0))


def get_team_points(x, team):

    #home stats    
    home_df = df[(df.home_team == team) & (df.match_day < x.match_day) & (df.season == x.season)]

    home_points = home_df.h_match_points.sum()

    home_wins = len(home_df[home_df['winner'] == 'HOME_TEAM'])
    home_draws = len(home_df[home_df['winner'] == 'DRAW'])
    home_losses = len(home_df[home_df['winner'] == 'AWAY_TEAM'])


    #away stats
    away_df = df[(df.away_team == team) & (df.match_day < x.match_day) & (df.season == x.season)]

    away_points = away_df.h_match_points.sum()
    away_wins = len(away_df[away_df['winner'] == 'AWAY_TEAM'])
    away_draws = len(away_df[away_df['winner'] == 'DRAW'])
    away_losses = len(away_df[away_df['winner'] == 'HOME_TEAM'])

    #total stats
    total_points = home_points + away_points
    total_wins = home_wins + away_wins
    total_draws = home_draws + away_draws
    total_losses = home_losses + away_losses

    #home points in last 3 games
    h_points = home_df[(home_df.match_day >= (x.match_day - 3)) & (home_df.match_day < x.match_day)]
    home_l_points = h_points.h_match_points.sum()

    #away points in last 3 games
    a_points = away_df[(away_df.match_day >= (x.match_day - 3)) & (away_df.match_day < x.match_day)]
    away_l_points = a_points.a_match_points.sum()

    #total points in last 3 games
    total_l_points = home_l_points + away_l_points

    return total_points , home_points , away_points, total_l_points, total_wins, total_draws, total_losses


def get_team_goals(x, team):

    #get team home goals made and suffered
    home_df = df[(df.match_day < x.match_day) & (df.home_team == team) & (df.season == x.season)]
    home_goals = home_df['home_score'].sum()
    home_goals_suffered = home_df['away_score'].sum()      

    #get team away goals
    away_df = df[(df.match_day < x.match_day) & (df.away_team == team) & (df.season == x.season)]
    away_goals = away_df['away_score'].sum()    
    away_goals_suffered = away_df['home_score'].sum()     

    #total goals made
    total_goals = home_goals + away_goals

    #total goals suffered
    total_goals_suffered = home_goals_suffered + away_goals_suffered

    return total_goals, total_goals_suffered


df.groupby(['season', 'match_day', 'home_team']).min()[['ht_total_points', 'ht_goals', 'ht_goals_sf']].reset_index()


def get_rank(x):

    #get current rank
    temp_df = df[(df.season == x.season) & (df.match_day == x.match_day)]

    home_df = temp_df.groupby(['season', 'match_day', 'home_team']).min()[['ht_total_points', 'ht_goals', 'ht_goals_sf']].reset_index()
    home_df.columns = ['season', 'match_day', 'team', 'total_points', 'goals', 'goals_sf']

    away_df = temp_df.groupby(['season', 'match_day', 'away_team']).min()[['at_total_points', 'at_goals', 'at_goals_sf']].reset_index()
    away_df.columns = ['season', 'match_day', 'team', 'total_points', 'goals', 'goals_sf']

    rank_df = pd.concat([home_df, away_df], ignore_index = True)
    rank_df['goals_df'] = rank_df.goals - rank_df.goals_sf
    rank_df = rank_df.sort_values(by = ['total_points', 'goals_sf', 'goals'], ascending = False)
    rank_df['rank'] = rank_df.total_points.rank(method = 'first', ascending = False).astype(int)

    home_team_rank = rank_df[rank_df.team == x.home_team].min()['rank']
    away_team_rank = rank_df[rank_df.team == x.away_team].min()['rank']
    
    #last season rank
    if x.season != 2018:
        temp_df = df[(df.season == (x.season - 1)) & (df.match_day <= 38)]

        home_df = temp_df.groupby(['season', 'home_team']).sum()[['h_match_points', 'home_score', 'away_score']].reset_index()
        home_df.columns = ['season', 'team', 'total_points', 'goals', 'goals_sf']

        away_df = temp_df.groupby(['season', 'away_team']).sum()[['a_match_points', 'away_score', 'home_score']].reset_index()
        away_df.columns = ['season', 'team', 'total_points', 'goals', 'goals_sf']

        rank_df = pd.concat([home_df, away_df], ignore_index = True)
        rank_df = rank_df.groupby(['season', 'team']).sum().reset_index()
        rank_df['goals_df'] = rank_df.goals - rank_df.goals_sf
        rank_df = rank_df.sort_values(by = ['total_points', 'goals_df', 'goals'], ascending = False)
        rank_df['rank'] = rank_df.total_points.rank(method = 'first', ascending = False).astype(int)

        home_team_last_rank = rank_df[rank_df.team == x.home_team].min()['rank']
        away_team_last_rank = rank_df[rank_df.team == x.away_team].min()['rank']
        
    else:
        home_team_last_rank = 0
        away_team_last_rank = 0

    return home_team_rank, home_team_last_rank,  away_team_rank, away_team_last_rank

#points so far,  points in latest 3 games and general stats
df[['ht_total_points', 'ht_home_points','ht_away_points', 'ht_ls_points',
     'ht_wins', 'ht_draws', 'ht_losses']] = pd.DataFrame(
    df.apply(
        lambda x: get_team_points(x, x.home_team), axis = 1).to_list(), index = df.index)

df[['at_total_points', 'at_home_points','at_away_points', 'at_ls_points',
     'at_wins', 'at_draws', 'at_losses']] = pd.DataFrame(
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
df[['ht_ranking', 'ht_last_ranking', 'at_ranking', 'at_last_ranking']] = pd.DataFrame(
    df.apply(
        lambda x: get_rank(x), axis = 1).to_list(), index = df.index)



#last season's position at current round
#history between two teams
#streak of wins
#streak of draws
#streak of losses
#days between last game
#result between last game of the teams
#maybe create my own metric to give weight to the teams


#amount of home w
#amount of away w
#amount of home d
#amount of away d
#amount of home l
#amount of away l

