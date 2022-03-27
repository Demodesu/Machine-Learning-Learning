#build a simple model
#but line doesn't accurately describe all data points -> underfit
#draw a line that exactly passes through all data points
#but equation is complicated and is bad -> overfit
#maybe make the line like a curve -> generalize points very well -> balanced fit
#how to reduce overfitting?
#make higer order variables close to 0 -> shrink parameters
#find MSE of equation 
#add adjustment -> higher theta, more error -> L2 regularization
#L1 use absolute

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

from sklearn import datasets

os.chdir("E:\Gun's stuff\Machine Learning\l1l2regularization")

import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv('Melbourne_housing_FULL.csv')

dataset.nunique() #print unique
dataset.shape #get how many rows and columns

cols_to_use = ['Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Regionname', 'Propertycount', 
               'Distance', 'CouncilArea', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Price']

dataset = dataset[cols_to_use]

dataset.isna().sum() #find NA values

cols_to_fill_zero = ['Propertycount','Distance','Bedroom2','Bathroom','Car']

dataset[cols_to_fill_zero] = dataset[cols_to_fill_zero].fillna(0) #fill NA with 0

dataset['Landsize'] = dataset['Landsize'].fillna(dataset.Landsize.mean()) #fill in NA with mean
dataset['BuildingArea'] = dataset['BuildingArea'].fillna(dataset.BuildingArea.mean()) #fill in NA with mean

dataset.dropna(inplace=True) #drop remaining random columns

#change categorical variables to numbers
#one hot encoding

dataset = pd.get_dummies(dataset, drop_first=True) #avoid dummy var trap (drops first column)

X = dataset.drop('Price', axis=1)
y = dataset['Price']

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=2)

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(train_X, train_y)
reg.score(test_X, test_y) #percent is low!
reg.score(train_X, train_y) #percent is high -> over fitting

from sklearn import linear_model
lasso_reg = linear_model.Lasso(alpha=50, max_iter=100, tol=0.1)
lasso_reg.fit(train_X, train_y)
lasso_reg.score(test_X, test_y) 
lasso_reg.score(train_X, train_y) #score is better

from sklearn.linear_model import Ridge
ridge_reg= Ridge(alpha=50, max_iter=100, tol=0.1)
ridge_reg.fit(train_X, train_y)
ridge_reg.score(test_X, test_y)
ridge_reg.score(train_X, train_y)
