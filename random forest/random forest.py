#predict based on certain features in a decision tree
#multiply decision trees from single data set?
#use one data set and separate into a batch of random samples
#and build decision tree for each of them
#forms a 'forest'
#put one thing to predict and they come out with different decisions
#use majority vote to predict
#example-> ask friend to recommend a drink
#1 friend said no 2 friends said yes
#decision is yes
#using random forest to classify digits

import pandas as pd
from sklearn.datasets import load_digits

digits = load_digits()

import matplotlib.pyplot as plt
plt.gray()
for i in range(4):
    plt.matshow(digits.images[i])

df = pd.DataFrame(digits.data) #digits.data is 1D array of numbers in 8x8

df['target'] = digits.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop(['target'],axis='columns'),digits.target,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier #ensemble uses multiple algorithms to predict outcome
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train,y_train) #Gini, estimators = number of decision trees we use, the more the more accurate
model.score(X_test,y_test)

y_predicted = model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predicted)

import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot = True)
plt.xlabel('predicted')
plt.ylabel('truth')
plt.show()


