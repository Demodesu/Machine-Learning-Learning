#https://www.youtube.com/watch?v=9yl6-HEY7_s&list=PLeo1K3hjS3uvCeTYTeyfe0-rN5r8zn9rw&index=6
#expand problem for nearby towns
#build a predictor function to predict price of a home
#1 -> 3400 sqr ft area in west windsor
#2 -> 2800 sqr ft area in robbinsville
#how to handle text data in numeric model?
#using integer encoding (change to numbers)
#categorical variables -> nominal and ordinal
#name of towns is in nominal
#so we use one hot encoding instead
#create new column for each category
#assign 1 and 0 for each, also known as dummy variables

import pandas as pd
import numpy as np
import os

os.chdir("E:\Gun's stuff\Machine Learning\dummy variables and one hot encoding")

df = pd.read_csv('homeprices.csv')

#print(pd.get_dummies(df.town))

    #create dummies

dummies = pd.get_dummies(df.town)

    #merge with original df

merged = pd.concat([df,dummies],axis='columns')

    #rule of thumb -> drop one dummy variable column always

final = merged.drop(['town','west windsor'],axis = 'columns')

from sklearn.linear_model import LinearRegression

model = LinearRegression()

X = final.drop('price',axis='columns')

Y = final.price

model.fit(X,Y)

#print(model.predict([[2800,0,1]])) #2800 sqr ft area in robbinsville

#print(model.predict([[3400,0,0]])) #3400 sqr ft area in west windsor

#print(model.score(X,Y)) #model accuracy

    #using from sklearn

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

dfle = df.copy()

dfle.town = le.fit_transform(dfle.town) #label encode is assigning a number to something

X = dfle[['town','area']].values #turn to 2D array

Y = dfle.price

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ohe = ColumnTransformer([('encoder',OneHotEncoder(),[0])],remainder='passthrough') #specify 0th column as categorical features

X = ohe.fit_transform(X)

X = X[:,1:] #drop 0th column

model.fit(X,Y)

print(model.predict([[0,0,3400]])) #change the position of the columns so that price is at the back