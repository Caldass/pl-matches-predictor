import pandas as pd
import numpy as np
import time
import os
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

#directories
BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
DATA_DIR = os.path.join(BASE_DIR, 'feature_eng', 'data', 'ft_df.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'heroku', 'models')

df = pd.read_csv(DATA_DIR)

#dropping columns one wouldn't have before an actual match
cols_to_drop = ['season', 'match_name','date', 'home_team', 'away_team', 'home_score', 'away_score',
                'h_match_points', 'a_match_points']

df.drop( columns = cols_to_drop, inplace = True)

#filling NAs
df.fillna(-33, inplace = True)

#turning the target variable into integers
df['winner'] = np.where(df.winner == 'HOME_TEAM', 2, np.where(df.winner == 'AWAY_TEAM', 1, 0))

#turning categorical into dummy vars
df_dum = pd.get_dummies(df)

np.random.seed(101)

X = df_dum.drop(columns = ['winner'], axis = 1)
y = df_dum.winner.values

#splitting into train and test set to check which model is the best one to work on
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#scaling features
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#creating models variable to iterate through each model and print result
models = [LogisticRegression(max_iter= 1000, multi_class = 'multinomial'),
RandomForestClassifier(), GradientBoostingClassifier(), KNeighborsClassifier()]

names = ['Logistic Regression', 'Random Forest', 'Gradient Boost', 'KNN']

#loop through each model and print train score and elapsed time
for model, name in zip(models, names):
    start = time.time()
    scores = cross_val_score(model, X_train, y_train ,scoring= 'accuracy', cv=5)
    print(name, ":", "%0.3f, +- %0.3f" % (scores.mean(), scores.std()), " - Elapsed time: ", time.time() - start)


#Creating loop to test which set of features is the best one for Logistic Regression

acc_results = []
n_features = []

#best classifier on training data
clf = LogisticRegression(max_iter = 1000, multi_class = 'multinomial')

for i in range(5, 40):
    rfe = RFE(estimator = clf, n_features_to_select = i, step=1)
    rfe.fit(X, y)
    X_temp = rfe.transform(X)

    np.random.seed(101)

    X_train, X_test, y_train, y_test = train_test_split(X_temp,y, test_size = 0.2)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    start = time.time()
    scores = cross_val_score(clf, X_train, y_train ,scoring= 'accuracy', cv=5)
    print(" Clf result :", "%0.3f, +- %0.3f" % (scores.mean(), scores.std()), 'N_features :', i)
    acc_results.append(scores.mean())
    n_features.append(i)

plt.plot(n_features, acc_results)
plt.ylabel('Accuracy')
plt.xlabel('N features')
plt.show()


#getting the best 13 features from RFE
rfe = RFE(estimator = clf, n_features_to_select = 13, step=1)
rfe.fit(X, y)
X_transformed = rfe.transform(X)

np.random.seed(101)
X_train, X_test, y_train, y_test = train_test_split(X_transformed,y, test_size = 0.2)

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


#getting column names
featured_columns = pd.DataFrame(rfe.support_,
                            index = X.columns,
                            columns=['is_in'])

featured_columns = featured_columns[featured_columns.is_in == True].index.tolist()

#column importances for each class
importances_d = pd.DataFrame(np.exp(rfe.estimator_.coef_[0]),
                            index = featured_columns,
                            columns=['coef']).sort_values('coef', ascending = False)

importances_a = pd.DataFrame(np.exp(rfe.estimator_.coef_[1]),
                            index = featured_columns,
                            columns=['coef']).sort_values('coef', ascending = False)

importances_h = pd.DataFrame(np.exp(rfe.estimator_.coef_[2]),
                            index = featured_columns,
                            columns=['coef']).sort_values('coef', ascending = False)


#tuning logistic regression
parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
 'fit_intercept': (True, False), 'solver' : ('newton-cg', 'sag', 'saga', 'lbfgs'), 'class_weight' : (None, 'balanced')}

gs = GridSearchCV(clf, parameters, scoring='accuracy', cv=3)
start = time.time()

#printing best fits and time elapsed
gs.fit(X_train,y_train)
print(gs.best_score_, gs.best_params_,  time.time() - start)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)


#testing models on unseen data 
tpred_lr = gs.best_estimator_.predict(X_test)
tpred_rf = rf.predict(X_test)
tpred_gb = gb.predict(X_test)
tpred_knn = knn.predict(X_test)

print(classification_report(y_test, tpred_lr, digits = 3))
print(classification_report(y_test, tpred_rf, digits = 3))
print(classification_report(y_test, tpred_gb, digits = 3))
print(classification_report(y_test, tpred_knn, digits = 3))



#function to get winning odd value in simulation dataset
def get_winning_odd(df):
    if df.winner == 2:
        result = df.h_odd
    elif df.winner == 1:
        result = df.a_odd
    else:
        result = df.d_odd
    return result

#creating dataframe with test data to simulate betting winnings with models

test_df = pd.DataFrame(scaler.inverse_transform(X_test),columns =  featured_columns)
test_df['tpred_lr'] = tpred_lr
test_df['tpred_rf'] = tpred_rf
test_df['tpred_gb'] = tpred_gb
test_df['tpred_knn'] = tpred_knn

test_df['winner'] = y_test
test_df['winning_odd'] = test_df.apply(lambda x: get_winning_odd(x), axis = 1)

test_df['lr_profit'] = (test_df.winner == test_df.tpred_lr) * test_df.winning_odd * 100
test_df['rf_profit'] = (test_df.winner == test_df.tpred_rf) * test_df.winning_odd * 100
test_df['gb_profit'] = (test_df.winner == test_df.tpred_gb) * test_df.winning_odd * 100
test_df['knn_profit'] = (test_df.winner == test_df.tpred_knn) * test_df.winning_odd * 100

investment = len(test_df) * 100

lr_return = test_df.lr_profit.sum() - investment
rf_return = test_df.rf_profit.sum() - investment
gb_return = test_df.gb_profit.sum() - investment
knn_return = test_df.knn_profit.sum() - investment

profit = (lr_return/investment * 100).round(2)

print(f'''Logistic Regression return: ${lr_return}

Random Forest return: ${rf_return}

Gradient Boost return:  ${gb_return}

KNN return:  ${knn_return} \n

Logistic Regression model profit percentage : {profit} %
''')


#retraining final model on full data
gs.best_estimator_.fit(X_transformed, y)

#Saving model and features
model_data = pd.Series( {
    'model': gs,
    'features': featured_columns
} )

#saving model
pickle.dump(model_data, open(os.path.join(MODEL_DIR, "lr_model.pkl"), 'wb'))