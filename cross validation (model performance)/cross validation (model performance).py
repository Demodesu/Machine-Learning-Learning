#how to know which model is the best?
#cross validation lets us evaluate model performance
#train and then use different data set to test compare with truth
#option 1 -> use all available data for training and testing on same dataset
#not good way of measuring because they have already seen the data
#option 2 -> split into training and testing data set
#good way to measure skills but maybe all data is of one type!
#might not perform well
#option 3 -> K fold cross validation
#example -> 100 samples into folds each containing 20 samples (0,1,2,3,4)
#use fold 0,1,2,3 to train, 4 to test, note down score
#use fold 4,0,1,2 to train, 3 to test, note down score
#use fold 3,4,0,1 to train, 2 to test, note down score
#repeat until all folds are tested (5 times in this case)
#finally average them out
#very good since there is variaty and we even out the score

from unicodedata import digit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.3)

lr = LogisticRegression(max_iter=10000)
lr.fit(X_train,y_train)
#print(lr.score(X_test,y_test))

svm = SVC()
svm.fit(X_train,y_train)
#print(svm.score(X_test,y_test))

rf = RandomForestClassifier(n_estimators=40)
rf.fit(X_train,y_train)
#print(rf.score(X_test,y_test))

#when we rerun the model, the score changes, and that is bad!
#train test split gathers random samples every time
#need to find average

from sklearn.model_selection import KFold

kf = KFold(n_splits=3) #3 folds

for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    #print(train_index,test_index) #this prints put the folds used to train vs test
    pass

def get_score(model,X_train,X_test,y_train,y_test):
    model.fit(X_train,y_train)
    return model.score(X_test,y_test)

#print(get_score(LogisticRegression(max_iter=10000),X_train,X_test,y_train,y_test))

from sklearn.model_selection import StratifiedKFold #similar to Kfold but is better by dividing classification uniformly ex. types of flowers
folds = StratifiedKFold(n_splits=3) #usually 10

scores_lr = []
scores_lvm = []
scores_rf = []

for train_index,test_index in kf.split(digits.data):
    X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], digits.target[train_index], digits.target[test_index]

    scores_lr.append(get_score(LogisticRegression(max_iter=10000),X_train, X_test, y_train, y_test))
    scores_lvm.append(get_score(SVC(),X_train, X_test, y_train, y_test))
    scores_rf.append(get_score(RandomForestClassifier(n_estimators=40),X_train, X_test, y_train, y_test))

print(scores_lr,scores_lvm,scores_rf)

#can also use

from sklearn.model_selection import cross_val_score

#print(cross_val_score(LogisticRegression(max_iter=10000),digits.data,digits.target))

