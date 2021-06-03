import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
DATA_DIR = os.path.join(BASE_DIR, 'feature_eng', 'data', 'ft_df.csv')

df = pd.read_csv(DATA_DIR)

cols_to_drop = ['season', 'date', 'status', 'home_team', 'away_team', 'home_score', 'away_score',
                'h_match_points', 'a_match_points', 'match_name']


df['winner'] = np.where(df.winner == 'HOME_TEAM', 2, np.where(df.winner == 'AWAY_TEAM', 1, 0))

df.fillna(-33, inplace = True)

#make merge here -------------------------------------
df_dum = pd.get_dummies(df, columns = ['ls_winner'] )

odds_df = df_dum.loc[(df_dum.season == 2020) & (df_dum.match_day >= 37)]
df_dum = df_dum.loc[~((df_dum.season == 2020) & (df_dum.match_day >= 37))]

odds_df.drop(columns = cols_to_drop, inplace = True)
df_dum.drop(columns = cols_to_drop, inplace = True)

df.shape
df_dum.shape
odds_df.shape

np.random.seed(101)

X = df_dum.drop(columns = ['winner'], axis = 1)
y = df_dum.winner.values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, shuffle = True)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

odds_df_ft = list(set(odds_df.columns) - set(['winner']))
odds_df[odds_df_ft] = scaler.fit_transform(odds_df[odds_df_ft])

#creating models variable to iterate through each model and print result
models = [LogisticRegression(max_iter = 1000),RandomForestClassifier(), GradientBoostingClassifier()]

names = ['Logistic Regression', 'Random Forest', 'Gradient Boost']

#loop through each model and print train score and elapsed time
for model, name in zip(models, names):
    start = time.time()
    scores = cross_val_score(model, X_train, y_train ,scoring= 'accuracy', cv=5)
    print(name, ":", "%0.3f, +- %0.3f" % (scores.mean(), scores.std()), " - Elapsed time: ", time.time() - start)