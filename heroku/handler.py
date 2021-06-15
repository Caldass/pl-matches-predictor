import pickle
import numpy as np
from flask import Flask, request
import pandas as pd
import os

#directories
BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
MODEL_DIR = os.path.join(BASE_DIR, 'heroku', 'models')

#loading model
model_data = pickle.load(open(os.path.join(MODEL_DIR, "lr_model.pkl"), 'rb'))

model = model_data['model']
features = model_data['features']

#instanciate flask
app = Flask(__name__)

#endpoint
@app.route('/predict', methods = ['POST'])
def predict():
    test_json = request.get_json()

    #get data
    if test_json:
        if isinstance(test_json, dict): #unique value
            df_raw = pd.DataFrame(test_json, index = [0])
        else:
            df_raw = pd.DataFrame(test_json, columns = test_json[0].keys())
    
    
    df_dum = df_raw.copy()
    
    #getting dummy ls_winner variable
    df = pd.get_dummies(df_dum)

    #checking if there are columns that don't exist in the input
    for f in features:
        if f not in df.columns:
            df[f] = 0

    #ordering columns
    df = df[features]

    #prediction
    pred = model.predict(df)
    df['prediction'] = pred

    #converting prediction into strings
    df['prediction'] = np.where(df.prediction == 2, 'Home team wins', np.where(df.prediction == 1, 'Away team wins', 'Draw'))

    return df.to_json(orient = 'records')

if __name__ == '__main__':
    #start flask
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port = port)

