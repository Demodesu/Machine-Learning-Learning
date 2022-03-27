#probability
#probability of event A knowing that B has already occured P(A/B)
#P(A/B) = P(B/A) * p(A) / p(B)
#P(queen/diamond) = P(diamond/queen) * P(queen) / P(diamond)
#data of titanic crash
#find survival rate
#P(survived/maleandclassandageandcabinandfate)
#naive because we assume that male class age cabin fate are independent of each other
#simple assumption to reduce calculation
#used in email spam detection, weather, face detection, character, etc.

import pandas as pd
import os

os.chdir("E:\Gun's stuff\Machine Learning\\naive bayes")

# df = pd.read_csv('titanic.csv')

# df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)

# target = df.Survived
# inputs = df.drop('Survived',axis='columns')

# dummies = pd.get_dummies(inputs.Sex) #separate sex into 2 columns

# inputs = pd.concat([inputs,dummies],axis='columns')

# inputs.drop('Sex',axis='columns',inplace=True)

# inputs.columns[inputs.isna().any()] #find NA -> in age

# #popular approach is to find the mean

# inputs.Age = inputs.Age.fillna(inputs.Age.mean())

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)

# from sklearn.naive_bayes import GaussianNB
# model = GaussianNB()

# model.fit(X_train,y_train)

# model.score(X_test,y_test)

# model.predict(X_test[:10])

# model.predict_proba(X_test[:10])

# #spam email detection
# #ham = good email
# #spam = bad email
# #train to detect spam

df = pd.read_csv('spam.csv')
print(df.groupby('Category').describe()) #tells us how many types are present

df['spam'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0) #applies a lambda function, if spam = 1, if not 0 and create a new column

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.Message,df.spam,test_size=0.25)

#convert text column into numbers using CountVectorizer -> turn words to count
#find unique words and make them features (0 or 1)

from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)

#print(X_train_count.toarray()[:3]) #features = the amount of unique words -> many columns!
#Bernoulli Naive Bayes -> assumes all features are binary (0 or 1) yes or no
#Multinomial Naive Bayes -> discrete data ex. ratings and count each word based on frequency to predict the class or label
#Guassian Naive Bayes -> data is continuous ex. iris data set where we can't represent features in terms of occurrences (normal distribution)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count,y_train)

emails = [
    'Hey, can we get together for dinner?',
    'Up to 20% discount on parking, do not miss this offer!'
]

emails_count = v.transform(emails)
print(model.predict(emails_count))

X_test_count = v.transform(X_test)
print(model.score(X_test_count,y_test))


#easier way to not need to transform lots of data

from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('ventorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

clf.fit(X_train,y_train) #directly feed text into model -> vectorizer first and then model

