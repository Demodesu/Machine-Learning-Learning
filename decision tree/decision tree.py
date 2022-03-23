#when we have a data set that is complex, we might have to set many boundaries
#salary more than 100k based on company, job, degree
#decision tree -> split into decision tree, sometimes inconclusive
#select ordering of decisions
#pure subset -> all samples are green, low entropy (randomness)
#high entropy -> 1 -> pure randomness
#LOW ENTROPY GETS HIGH INFORMATION GAIN!!!
#use company as first attribute
#Gini impurity -> impurity in data set
#ex. most samples are red, 1 is green

import pandas as pd
import os
from matplotlib import pyplot as plt

os.chdir("E:\Gun's stuff\Machine Learning\decision tree")

df = pd.read_csv('salaries.csv')

inputs = df.drop('salary_more_then_100k',axis='columns')
target = df['salary_more_then_100k']

#convert columns into numbers -> label encoder

from sklearn.preprocessing import LabelEncoder

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

    #assign number to labels

inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])

inputs_n = inputs.drop(['company','job','degree'],axis='columns')
#print(inputs_n)

from sklearn import tree

model = tree.DecisionTreeClassifier()

model.fit(inputs_n,target) #criterion = Gini, Entropy?

model.score(inputs_n,target) #score is 1 because we use the same data for training and testing and the data is simple

print(model.predict([[2,2,1]])) #0 means that salary less than 100k