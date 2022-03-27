#sepal width height of iris classification
#new data point
#nearest to cluster
#K = 3, 3 nearest data points
#if 3 nearest is verginica, then new data is most likely verginica
#K = 10, 10 nearest data points
#verginica 7, versicolor 3 -> new data verginica
#K = 20, 20 nearest data points
#verginica 7, versicolor 11, selosa 2 -> new data versicolor (misclassify)
#K should not be too high and not too low!

import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()


iris.feature_names
iris.target_names
df = pd.DataFrame(iris.data,columns=iris.feature_names)

df['target'] = iris.target
df.head()

df[df.target==1].head()

df[df.target==2].head()

df['flower_name'] =df.target.apply(lambda x: iris.target_names[x])

df0 = df[:50]
df1 = df[50:100]
df2 = df[100:] #split into 3 data frames

import matplotlib.pyplot as plt

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="blue",marker='.')

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],color="green",marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],color="blue",marker='.')

from sklearn.model_selection import train_test_split
X = df.drop(['target','flower_name'], axis='columns')
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
len(X_train)
len(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10) #try to figure out best value of K
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
knn.predict([[4.8,3.0,1.5,0.3]])

from sklearn.metrics import confusion_matrix #tells us which class got prediction right and wrong
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred) #first is truth, second is prediction

import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(7,5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

from sklearn.metrics import classification_report #used to look at final report
print(classification_report(y_test, y_pred))

