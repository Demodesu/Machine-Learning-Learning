#linear regression 
#home prices, weather, stock prices -> predicted value is continuous
#what if email is spam or not? will customer buy insurance? which part is going to win?
#categorical!!!
#we call this classification problems
#two types
#yes or no (binary classification)
#blue, purple, red, ... (multiclass classification)
#plot scatter plot and draw linear regression?
#plot graph > 0.5, likely to buy insurance or smt (0 = no insurance, 1 = has insurance)
#but results will be pretty bad, since data is close or not accurate
#use sigmoid or logit function
#sigmoid(z) = 1/(1+e^-z) e = 2.71828
#sigmoid function converts input into range 0 to 1
#how about if we feed a line equation into the sigmoid function?
#the equation becomes -> sigmoid(z) = 1/(1+e^-(m*x+b))
#turns straight line into sigmoid line

    #binary

import pandas as pd
import os
from matplotlib import pyplot as plt

os.chdir("E:\Gun's stuff\Machine Learning\logistic regression")

df = pd.read_csv('insurance_data.csv')

#plt.scatter(df.age,df.bought_insurance,marker='+',color='red')
#plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance,test_size=0.1,random_state=10)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train,y_train) #training for model

#print(y_test)
#print(model.predict(X_test))

model.score(X_test,y_test)

print(model.predict_proba(X_test)) #shows probability that test will be in one class vs another
print(X_test)

    #multiclass

#identify written digits recognition etc.

import matplotlib.pyplot as plt

from sklearn.datasets import load_digits

digits = load_digits()

dir(digits)

#print(digits.data[0]) #8x8

plt.gray()
#plt.matshow(digits.images[0])
#plt.show()

#print(digits.target[0:5])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.2)

#print(len(X_test), len(X_train))

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='lbfgs',max_iter=10000)

model.fit(X_train,y_train)

#print(model.score(X_test,y_test))

#plt.matshow(digits.images[67])
#plt.show()

#print(model.predict(digits.data[0:5]))

y_predicted = model.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predicted) #area outside i column is off, predict vs truth

import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot = True)
plt.xlabel('predicted')
plt.ylabel('truth')
plt.show()