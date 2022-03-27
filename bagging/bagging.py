#use multiple models to get opinions on a dataset
#reduces bias

#train multiple models and do prediction on multiple models
#combine results to get final result

#resampling with replacement to create small samples out of our dataset
#randomly pick data point, pick data point with random probability
#might get same sample again
#create n numbers of subsets
#on individual dataset train a model
#predict on all models 
#take majority vote

#benefit is that each model is weak learners, will not overfit

#also called bootstrap aggregation

#random forest algorithm sample rows and also columns
#train decision tree
#average results

#bagging can be any model
#bagged tree is each model is a tree

from random import random
import pandas as pd
import os

os.chdir("E:\Gun's stuff\Machine Learning\\bagging")

df = pd.read_csv('diabetes.csv')

df.isnull().sum() #find which columns that is null or NA

df.describe() #tells us basic statistics

#if there are outliers we need to get rid of them

df.Outcome.value_counts() #find the imbalance of diabetes/no diabetes

#major imbalance we should worry

X = df.drop('Outcome',axis='columns')
y = df.Outcome

#scaling

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.2,stratify=y,random_state=10) #stratify lets us maintain the ratio of outcomes

y_train.value_counts() #same ratio

from sklearn.tree import DecisionTreeClassifier #can overfit, high variance
from sklearn.model_selection import cross_val_score
scores = cross_val_score(DecisionTreeClassifier(),X,y,cv=5) #kfold 5 times
print(scores)
print(scores.mean())

from sklearn.ensemble import BaggingClassifier
bag_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0
) 
#100 models, 100 subsets, train in parallel, 80% of data as samples
#out of bag (oob) some value might be missed, can use that score to test model (test data set)
#oob is the accuracy used by testing values that don't appear to values that appear

bag_model.fit(X_train,y_train)
bag_model.oob_score_

bag_model.score(X_test,y_test) #better than scores.mean

bag_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0
) 
scores = cross_val_score(bag_model,X,y,cv=5)
scores.mean() #bag model gives more accuracy even for unstable classifier like decision tree

from sklearn.ensemble import RandomForestClassifier
cross_val_score(RandomForestClassifier(),X,y,cv=5)
scores.mean() #better performance