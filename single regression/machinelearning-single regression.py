#simple linear regression
#calculate delta, which is distance from a point to a regression line to give minimum error
#predict the prices of homes from sqr.ft area

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import os

    #find current directory

#print(os.getcwd()) 

    #change current directory

os.chdir("E:\Gun's stuff\Machine Learning\single regression")

    #load data frame

df = pd.read_csv('homeprices.csv')

    #regression

rg = linear_model.LinearRegression()
#training using fit(), train using only values
rg.fit(df[['area']].values, df.price.values)
#predict price at 3300 sqr.ft
#print(rg.predict([[3300]]))
#print the linear regression coeff. or slope
#print(rg.coef_)
#print the y intercept
#print(rg.intercept_)

    #predict some prices for areas

#d = pd.read_csv('areas.csv')
#d.head(3)

#p = rg.predict(d.values)
#create new column in data frame
#d['prices'] = p
#d.to_csv('predictions.csv')

    #plot

plt.scatter(df.area, df.price, color = 'red', marker = '+')
plt.xlabel('area(sqr.ft)')
plt.ylabel('price(US$)')
plt.plot(df.area, rg.predict(df[['area']].values), color = 'blue')
plt.show()
