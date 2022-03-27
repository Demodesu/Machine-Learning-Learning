#overfit, underfit, balanced
#pass through all, pass too little, perfect

#overfit
#error is 0, test error is high
#test error varies greatly -> high variance (based on randomly selected training data)

#underfit
#simple model, linear equation
#cannot capture all points in data
#training error is high, testing error is low
#low variance 
#high bias (train error) ability to capture pattern in training dataset

#balanced
#accurately describes model
#train and test error is low even when we change samples
#low variance, low bias

#cross validation, regularization, dimensionality reduction, ensemple techniques -> ways to get balanced fit
#kfold, l1l2, PCA, bagging/boosting 