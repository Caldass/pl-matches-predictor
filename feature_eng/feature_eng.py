import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
DATA_DIR = os.path.join(BASE_DIR, 'get_data', 'data')
FT_DIR = os.path.join(BASE_DIR, 'feature_eng', 'data')

df = pd.read_csv(os.path.join(DATA_DIR, 'matches.csv'))

#checking column types and null values
df.info()
df.isna().sum()

#setting season to be the start year of a season
df['season'] = df.season.str.split('/').str[0]

#creating home and away score
df[['home_team', 'away_team']] = df.match_name.str.split(' - ', expand = True)

#creating home and away score
df[['home_score', 'away_score']] = df.result.str.split(':', expand = True)

df['winner'] = np.where(df.home_score > df.away_score, 'HOME_TEAM', np.where(df.away_score > df.home_score, 'AWAY_TEAM', 'DRAW'))

df.drop(columns = 'result', inplace = True)

to_int = ['season','home_score', 'away_score']
to_float = ['a_odd', 'd_odd', 'h_odd']


#turning columns into integers and floats
for col in to_int:
    df[col] = df[col].astype(int)

for col in to_float:
    df[col] = df[col].str.replace('-', '0')
    df[col] = df[col].astype(float)


#changing from date to datetime
df['date'] = pd.to_datetime(df.date)

#home points made in each match
df['h_match_points'] = np.where(df['winner'] == 'HOME_TEAM', 3 , np.where(df['winner'] == 'DRAW',1, 0))

#away points made in each match
df['a_match_points'] = np.where(df['winner'] == 'AWAY_TEAM', 3 , np.where(df['winner'] == 'DRAW',1, 0))


cols_order = ['season', 'date', 'match_name', 'home_team', 'away_team', 'winner', 'home_score', 'away_score',
                'h_odd', 'd_odd', 'a_odd', 'h_match_points', 'a_match_points']

df = df[cols_order]

df.head()

def get_rank(x, team, delta_year):
    full_season_df = df[(df.season == (x.season - delta_year))]

    full_home_df = full_season_df.groupby(['home_team']).sum()[['h_match_points', 'home_score', 'away_score']].reset_index()
    full_home_df.columns = ['team', 'points', 'goals', 'goals_sf']

    full_away_df = full_season_df.groupby(['away_team']).sum()[['a_match_points', 'away_score', 'home_score']].reset_index()
    full_away_df.columns = ['team', 'points', 'goals', 'goals_sf']

    rank_df = pd.concat([full_home_df, full_away_df], ignore_index = True)
    rank_df['goals_df'] = rank_df.goals - rank_df.goals_sf
    rank_df = rank_df.groupby(['team']).sum().reset_index()
    rank_df = rank_df.sort_values(by = ['points', 'goals_df', 'goals'], ascending = False)
    rank_df['rank'] = rank_df.points.rank(method = 'first', ascending = False).astype(int)

    team_rank = rank_df[rank_df.team == team].min()['rank']

    return team_rank

def get_match_stats(x, team):
    #home df filter    
    home_df = df[(df.home_team == team) & (df.date < x.date) & (df.season == x.season)]

    #home df filter
    away_df = df[(df.away_team == team) & (df.date < x.date) & (df.season == x.season)]

    #points
    home_table = home_df.groupby(['date']).sum()[['h_match_points', 'home_score', 'away_score']].reset_index()
    home_table.columns = ['date', 'points', 'goals', 'goals_sf']
    home_table['goals_df'] = home_table.goals - home_table.goals_sf
    home_table['host'] = 'home'

    away_table = away_df.groupby(['date']).sum()[['a_match_points', 'away_score', 'home_score']].reset_index()
    away_table.columns = ['date', 'points', 'goals', 'goals_sf']
    away_table['goals_df'] = away_table.goals - away_table.goals_sf
    away_table['host'] = 'away'

    full_table = pd.concat([home_table, away_table], ignore_index = True)
    full_table = full_table.sort_values('date', ascending = True)

    #get streaks
    full_table['start_of_streak'] = full_table.points.ne(full_table.points.shift())
    full_table['streak_id'] = full_table['start_of_streak'].cumsum()
    full_table['streak_counter'] = full_table.groupby('streak_id').cumcount() + 1

    #make exponentially weighted average
    full_table['w_avg_points'] = full_table.points.ewm(span=3, adjust=False).mean()
    full_table['w_avg_goals'] = full_table.goals.ewm(span=3, adjust=False).mean()
    full_table['w_avg_goals_sf'] = full_table.goals_sf.ewm(span=3, adjust=False).mean()

    streak_table = full_table[full_table.date == full_table.date.max()]

    if streak_table.points.min() == 3:
        win_streak = streak_table.streak_counter.sum()
        loss_streak = 0
        draw_streak = 0
    elif streak_table.points.min() == 0:
        win_streak = 0
        loss_streak = streak_table.streak_counter.sum()
        draw_streak = 0
    else:
        win_streak = 0
        loss_streak = 0
        draw_streak = streak_table.streak_counter.sum()
    
    home_points = home_table.points.sum()
    home_goals = home_table.goals.sum()
    home_goals_sf = home_table.goals_sf.sum()
    home_wins = len(home_table[home_table.points == 3])
    home_draws = len(home_table[home_table.points == 1])
    home_losses = len(home_table[home_table.points == 0])


    away_points = away_table.points.sum()
    away_goals = away_table.goals.sum()
    away_goals_sf = away_table.goals_sf.sum()
    away_wins = len(away_table[away_table.points == 3])
    away_draws = len(away_table[away_table.points == 1])
    away_losses = len(away_table[away_table.points == 0])

    #total points stats
    total_points = home_points + away_points
    total_goals = home_goals + away_goals
    total_goals_sf = home_goals_sf + away_goals_sf
    total_wins = home_wins + away_wins
    total_draws = home_draws + away_draws
    total_losses = home_losses + away_losses
    
    #getting data for a given delta
    full_table_delta = full_table[full_table.date.isin(full_table.date[-3:])]

    home_l_points = full_table_delta[full_table_delta.host == 'home'].points.sum()
    away_l_points = full_table_delta[full_table_delta.host == 'away'].points.sum()

    #total metric in given delta averaged
    total_l_points = (home_l_points + away_l_points)/3
    total_l_goals = (home_goals + away_goals)/3
    total_l_goals_sf = (home_goals_sf + away_goals)/3

    total_l_w_avg_points = full_table[full_table.date.isin(full_table.date[-1:])].w_avg_points.sum()
    total_l_w_avg_goals = full_table[full_table.date.isin(full_table.date[-1:])].w_avg_goals.sum()
    total_l_w_avg_goals_sf = full_table[full_table.date.isin(full_table.date[-1:])].w_avg_goals_sf.sum()

    return total_points, total_l_points, total_l_w_avg_points, total_goals, total_l_goals, total_l_w_avg_goals, total_goals_sf, total_l_goals_sf, total_l_w_avg_goals_sf, total_wins, total_draws, total_losses, win_streak, loss_streak, draw_streak

def get_days_ls_match(x, team):

    #filtering last game of the team and getting date
    last_date = df[(df.date < x.date) & (df.season == x.season) & (df.match_name.str.contains(team))].date.max()

    days = (x.date - last_date)/np.timedelta64(1,'D')

    return days

def get_ls_winner(x):
    temp_df = df[(df.date < x.date) & (df.match_name.str.contains(x.home_team)) & (df.match_name.str.contains(x.away_team))]
    temp_df = temp_df[temp_df.date == temp_df.date.max()]
    
    #checking if there was a previous match
    if len(temp_df) == 0:
        result = None
    elif temp_df.winner.all() == 'DRAW':
        result = 'DRAW'
    elif temp_df.home_team.all() == x.home_team:
        result = temp_df.winner.all()
    else:
        if temp_df.winner.all() == 'HOME_TEAM':
            result = 'HOME_TEAM'
        else:
            result = 'AWAY_TEAM'
    
    return result

def create_main_cols(x, team):

    #get current and last delta (years) rank
    team_rank = get_rank(x, team, 0)
    ls_team_rank = get_rank(x, team, 1)

    #get main match stats
    total_points, total_l_points, total_l_w_avg_points, total_goals, total_l_goals, total_l_w_avg_goals, total_goals_sf, total_l_goals_sf, total_l_w_avg_goals_sf, total_wins, total_draws, total_losses, win_streak, loss_streak, draw_streak = get_match_stats(x, team)

    #get days since last match
    days = get_days_ls_match(x, team)    

    return team_rank, ls_team_rank, days, total_points, total_l_points, total_l_w_avg_points, total_goals, total_l_goals, total_l_w_avg_goals, total_goals_sf, total_l_goals_sf, total_l_w_avg_goals_sf, total_wins, total_draws, total_losses, win_streak, loss_streak, draw_streak

cols = ['_rank', '_ls_rank', '_days_ls_match', '_points',
 '_l_points', '_l_wavg_points', '_goals', '_l_goals', '_l_wavg_goals', '_goals_sf', '_l_goals_sf', '_l_wavg_goals_sf','_wins', '_draws', '_losses', '_win_streak', '_loss_streak', '_draw_streak']

ht_cols = ['ht' + col for col in cols]
at_cols = ['at' + col for col in cols]

#gets main cols for home and away team
df[ht_cols] = pd.DataFrame(
    df.apply(
        lambda x: create_main_cols(x, x.home_team), axis = 1).to_list(), index = df.index)

df[at_cols] = pd.DataFrame(
    df.apply(
        lambda x: create_main_cols(x, x.away_team), axis = 1).to_list(), index = df.index)        

#result between last game of the teams
df['ls_winner'] = df.apply(lambda x: get_ls_winner(x), axis = 1)

#saving data
df.to_csv(os.path.join(FT_DIR, 'ft_df.csv'), index = False)