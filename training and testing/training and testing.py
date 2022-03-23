#split the data set into two parts
#ex. training 80% of data, testing 20% of data
#we do this because we need to test data that the model hasn't seen before

import pandas as pd
import numpy as np
import os

os.chdir("E:\Gun's stuff\Machine Learning\ml training and testing")

df = pd.read_csv('carprices.csv')

X = df[['Mileage','Age(yrs)']]

y = df['Sell Price($)']

#use linear since there is clear relationship

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10) #testing 20% #random_state = 10 uses the same values everytime

print(len(X_train), len(X_test))

from sklearn.linear_model import LinearRegression

clf = LinearRegression()

clf.fit(X_train, y_train)

print(clf.predict(X_test)) #using the test samples to calculate

print(y_test) #comparing the test sample results

print(clf.score(X_test,y_test)) #compare accuracy