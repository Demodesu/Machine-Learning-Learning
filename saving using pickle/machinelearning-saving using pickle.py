#2 steps for machine learning 
#1 -> training the model 
#2 -> asking the trained model a question
#size of training data is huge -> more data, more accurate
#often in GB of training data, lots of time to train
#save the model to not train it again

import pandas as pd
import numpy as np
from sklearn import linear_model
import os

os.chdir("E:\Gun's stuff\Machine Learning\saving using pickle")

df = pd.read_csv('homeprices.csv')
df.head

model = linear_model.LinearRegression()
model.fit(df[['area']], df.price)

    #allows us to save files

import pickle

with open('model_pickle','wb') as f: #wb to save as binary
    pickle.dump(model,f)

with open('model_pickle', 'rb') as f: #rb to read binary
    pickle_model = pickle.load(f)

print(pickle_model.predict([[5000]]))

import joblib

joblib.dump(model, 'model_joblib')

joblib_model = joblib.load('model_joblib')

print(joblib_model.predict([[5000]]))