#multi regression predictions
#predicting something with multiple variables
#predicting the price with area, bedrooms, and age
#filling in missing data points
#linear relationship between different variables
#example eq. price = m1*area + m2*bedrooms + m3*age + c 
#independent variable = features

import pandas as pd
import numpy as np
import math
from sklearn import linear_model
import os

    #change current directory

os.chdir("E:\Gun's stuff\Machine Learning\multiple regression")

    #load data frame

df = pd.read_csv('homeprices.csv')

    #handle missing data!!

median_bedrooms = math.floor(df.bedrooms.median())
#fill missing data
df.bedrooms = df.bedrooms.fillna(median_bedrooms)

    #regression

reg = linear_model.LinearRegression()
#train
reg.fit(df[['area', 'bedrooms', 'age']], df.price) #first argument is independent variables, second is the target variable
#print(reg.coef_, reg.intercept_)
print(reg.predict([[3000,3,40]]))


