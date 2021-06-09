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
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt



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

selected_features = ['h_odd', 'd_odd', 'a_odd', 'ht_rank', 
        'ht_days_ls_match', 'ht_l_points',
        'ht_l_goals', 'ht_goals_sf',
        'ht_l_wavg_goals_sf', 
       'ht_win_streak', 'ht_loss_streak',
       'at_rank', 'at_days_ls_match','at_l_points',
       'at_l_goals', 'at_goals_sf',
        'at_l_wavg_goals_sf', 
       'at_win_streak', 'at_loss_streak', 'at_l_wavg_goals', 'ht_l_wavg_goals']
       

X = df_dum.drop(columns = ['winner'], axis = 1)
X = X[selected_features]
y = df_dum.winner.values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 101)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#creating models variable to iterate through each model and print result
models = [LogisticRegression(max_iter= 1000, multi_class = 'multinomial'),RandomForestClassifier(), GradientBoostingClassifier()]

names = ['Logistic Regression', 'Random Forest', 'Gradient Boost']

#loop through each model and print train score and elapsed time
for model, name in zip(models, names):
    start = time.time()
    scores = cross_val_score(model, X_train, y_train ,scoring= 'accuracy', cv=5)
    print(name, ":", "%0.3f, +- %0.3f" % (scores.mean(), scores.std()), " - Elapsed time: ", time.time() - start)

np.random.seed(101)

X = df_dum.drop(columns = ['winner'], axis = 1)
y = df_dum.winner.values

clf = LogisticRegression(max_iter = 1000, multi_class = 'multinomial')

rfe = RFE(estimator = clf, n_features_to_select = 13, step=1)
rfe.fit(X, y)
X, y = rfe.transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


scores = cross_val_score(clf, X_train, y_train ,scoring= 'accuracy', cv=5)
print(" Clf result :", "%0.3f, +- %0.3f" % (scores.mean(), scores.std()))

a = pd.DataFrame(selector.support_,
                                   index = X.columns,
                                    columns=['importance']).sort_values('importance', ascending = True)

a[a.importance == True].index.tolist()

#RFE --------------------------------------------------------------------------
acc_results = []
n_features = []

for i in range(5, 40):
    X = df_dum.drop(columns = ['winner'], axis = 1)
    y = df_dum.winner.values

    selector = RFE(LogisticRegression(max_iter= 1000, multi_class = 'multinomial'), n_features_to_select = i, step=1)
    selector.fit_transform(X, y)


    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    clf = LogisticRegression(max_iter = 1000, multi_class = 'multinomial')


    start = time.time()
    scores = cross_val_score(clf, X_train, y_train ,scoring= 'accuracy', cv=5)
    print(" Clf result :", "%0.3f, +- %0.3f" % (scores.mean(), scores.std()), 'N_features :', i)
    acc_results.append(scores.mean())
    n_features.append(i)

plt.plot(n_features, acc_results)
plt.show()


for i in range(X.shape[1]):
	print('Column: %d, Selected %s, Rank: %.3f' % (i, selector.support_[i], selector.ranking_[i]))

a = pd.DataFrame(selector.support_,
                                   index = X.columns,
                                    columns=['importance']).sort_values('importance', ascending = True)

a[a.importance == True].index.tolist()



#end of rfe ---------------------------------------


clf = LogisticRegression(max_iter = 1000, multi_class = 'multinomial')
clf.fit(X_train, y_train)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

result = clf.predict(X_test)

print(classification_report(y_test, result, digits = 3))


#feature importances

imp = clf.feature_importances_


feature_importances = pd.DataFrame(np.exp(clf.coef_[1]),
                                   index = X.columns,
                                    columns=['importance']).sort_values('importance', ascending = False)

feature_importances

clf.classes_
clf.coef_[2]





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

retorno = test_df.profit.sum()
investimento = len(test_df) * 100

lucro = retorno - investimento

lucro/investimento




test_df[test_df.pred == 0]








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