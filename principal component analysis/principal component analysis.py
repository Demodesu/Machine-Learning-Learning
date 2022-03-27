#find the factor that is MOST important that affects the result
#identify the features that are important

#example digit recognition
#some pixels don't have any role in features anymore
#get rid of them? training is faster, data visualization becomes easier

#principal component analysis is figuring the most important factor
#plot corner pixel and central pixel
#corner pixel doesn't play much effect

#two lines PC1 and PC2
#one line covers most of the variance and the other is perpendicular

#SCALE FEATURES first -> it might be skewed if not scaled
#ACCURACY might DROP -> 100 features might be important but we get rid of them

#PCA allows us to reduce dimensions (features)
#dimensional curse -> too many dimensions

import pandas as pd
from sklearn.datasets import load_digits

dataset = load_digits()
dataset.keys() #get the keys to use

dataset.data[0] #1D array 
dataset.data[0].reshape(8,8) #2D array #features

from matplotlib import pyplot as plt

plt.gray()
plt.matshow(dataset.data[0].reshape(8,8))
plt.show()

import numpy as np
np.unique(dataset.target) #find unique values #class

df = pd.DataFrame(dataset.data,columns=dataset.feature_names)

df.describe() #find count, mean, max, min , etcs

X = df
y = dataset.target

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=30) #random state for reusability

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
model.score(X_test,y_test)

from sklearn.decomposition import PCA
pca = PCA(0.95) #retain 95% of most useful features
X_pca = pca.fit_transform(X)
X_pca.shape #find the rows and columns

pca.explained_variance_ratio_ #tells us which one gives us most variance

X_pca_train, X_pca_test, y_train, y_test = train_test_split(X_pca,y,test_size=0.2,random_state=30)

model = LogisticRegression(max_iter=1000)
model.fit(X_pca_train,y_train)
model.score(X_pca_test,y_test)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_pca.shape #gives us 2 most important features, but not enough information


