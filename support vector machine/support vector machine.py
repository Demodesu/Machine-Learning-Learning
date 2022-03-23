#using petal and sepal size to determine species of iris
#higher margin is better
#maximize margin between nearby data points and sign
#points are called support vectors
#2D -> boundary is line
#3D -> plane
#nD? -> hyper plane
#SVM lets us do that
#boundary considers points only VERY close to it -> high gamma
#boundary considers far away points as well -> low gamma (sometimes we get problem with accuracy but faster compute time)
#complex data set -> line might overfit and can be zig zag -> high regularization
#take some classifications errors, line is smoother -> low regularization
#make Z plane -> X^2 + Y^2
#Z is called kernal to draw boundary easier

import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()

iris.feature_names #things like sepal length, width etc.

df = pd.DataFrame(iris.data,columns=iris.feature_names)

df['target'] = iris.target #the species

iris.target_names #3 types of species determined by 4 features

df[df.target==1].head() #find target name = 1

df['flower_name'] = df.target.apply(lambda x: iris.target_names[x]) #for each value in target column, return index in target names

from matplotlib import pyplot as plt

df0 = df[df.target==0]
df1 = df[df.target==1]
df2 = df[df.target==2]

fig, axs = plt.subplots(2)
axs[0].scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],marker = '+')
axs[0].scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],marker = '+',color='red')
axs[1].scatter(df0['petal length (cm)'],df0['petal width (cm)'],marker = '+')
axs[1].scatter(df1['petal length (cm)'],df1['petal width (cm)'],marker = '+',color='red')

plt.show()

from sklearn.model_selection import train_test_split

X = df.drop(['target', 'flower_name'],axis='columns')
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

from sklearn.svm import SVC

model = SVC(C = 10) #c = regularization, can tune other parameters, gamma, kernal?

model.fit(X_train,y_train)

model.score(X_test, y_test)

