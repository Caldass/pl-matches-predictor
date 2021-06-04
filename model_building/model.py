import pandas as pd
import numpy as np
import time
import os
from pandas.io.formats.format import TextAdjustment
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
DATA_DIR = os.path.join(BASE_DIR, 'feature_eng', 'data', 'ft_df.csv')

df = pd.read_csv(DATA_DIR)

df.info()

cols_to_drop = ['season', 'match_name','date', 'home_team', 'away_team', 'home_score', 'away_score',
                'h_match_points', 'a_match_points']

df.drop( columns = cols_to_drop, inplace = True)

df['winner'] = np.where(df.winner == 'HOME_TEAM', 2, np.where(df.winner == 'AWAY_TEAM', 1, 0))

df.fillna(-33, inplace = True)

df_dum = pd.get_dummies(df)

np.random.seed(101)

X = df_dum.drop(columns = ['winner'], axis = 1)
y = df_dum.winner.values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#creating models variable to iterate through each model and print result
models = [LogisticRegression(max_iter= 1000),RandomForestClassifier(), GradientBoostingClassifier()]

names = ['Logistic Regression', 'Random Forest', 'Gradient Boost']

#loop through each model and print train score and elapsed time
for model, name in zip(models, names):
    start = time.time()
    scores = cross_val_score(model, X_train, y_train ,scoring= 'accuracy', cv=5)
    print(name, ":", "%0.3f, +- %0.3f" % (scores.mean(), scores.std()), " - Elapsed time: ", time.time() - start)

clf = LogisticRegression(max_iter = 1000)
clf.fit(X_train, y_train)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

result = clf.predict(X_test)

print(accuracy_score(y_test, result))

pqp = df_dum.columns
imp = clf.feature_importances_


feature_importances = pd.DataFrame(clf.coef_[0],
                                   index = X.columns,
                                    columns=['importance']).sort_values('importance', ascending = False)

feature_importances


#getting proffits
def get_winning_odd(df):
    if df.winner == 2:
        result = df.h_odd
    elif df.winner == 1:
        result = df.a_odd
    else:
        result = df.d_odd
    return result

test_df = pd.DataFrame(scaler.inverse_transform(X_test),columns =  X.columns)
test_df['pred'] = result
test_df['winner'] = y_test

test_df['winning_odd'] = test_df.apply(lambda x: get_winning_odd(x), axis = 1)
test_df.head()

test_df['profit'] = (test_df.winner == test_df.pred) * test_df.winning_odd * 100

test_df['profit'] = np.where(test_df.profit == 0, -100, test_df.profit)
retorno = test_df.profit.sum()
investimento = len(test_df) * 100

lucro = retorno - investimento














'''
def get_prediction(df):
    if (df.h_odd < df.d_odd) & (df.h_odd < df.a_odd):
        result = 2
    elif (df.a_odd < df.d_odd) & (df.a_odd < df.h_odd):
        result = 1
    else:
        result = 0
    return result

df['teste'] = df.apply(lambda x: get_prediction(x), axis = 1)
df['pls'] = df.teste == df.winner

df.pls.mean()

df.to_excel('eohq.xlsx', index = False)'''