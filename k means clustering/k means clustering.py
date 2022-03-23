#we have a data set representing 2 different features and we want to identify clusters
#we don't know what we are looking for
#helps us identify clusters
#K is the amount of clusters (free variable)
#start with K centroids by putting at random place
#find the distance between a point to the centroids
#if it is closer to centroid 1, then it is red, etc.
#try to adjust the centroid of 2 clusters
#find the center between all points of and put centroid there
#repeat finding the distance
#repeat recalculating centroid
#until none of data points change cluster
#what is a good K?
#elbow method!
#start with some K
#compute sum of square error
#add all of sum of square error
#draw a plot y = SSE, x = K
#increase cluster, decrease error
#find the elbow (point at which slope is almost linear)

import os

os.chdir("E:\Gun's stuff\Machine Learning\k means clustering")

from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

df = pd.read_csv('income.csv')

# plt.scatter(df['Age'],df['Income($)']) #we can clearly see 3 clusters

# km = KMeans(n_clusters=3) #K is n_clusters
# #print(km) #prints the parameters that we can change

# y_predicted = km.fit_predict(df[['Age','Income($)']]) #automatically assigns value to number of arrays

# df['cluster'] = y_predicted

# df1 = df[df.cluster==0]
# df2 = df[df.cluster==1]
# df3 = df[df.cluster==2]

# plt.scatter(df1.Age,df1['Income($)'],color='green')
# plt.scatter(df2.Age,df2['Income($)'],color='red')
# plt.scatter(df3.Age,df3['Income($)'],color='blue')

#scale data imporperly -> not accurate! -> preprocessing

scalar = MinMaxScaler()
scalar.fit(df[['Income($)']])
df['Income($)'] = scalar.transform(df[['Income($)']])

scalar.fit(df[['Age']])
df[['Age']] = scalar.transform(df[['Age']])

#train scaled dataset

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age','Income($)']])

df['cluster'] = y_predicted

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='blue')

plt.xlabel('age')
plt.ylabel('income ($)')

#now data is nicely formed (clustered)

#print(km.cluster_centers_) #where the cluster centroids are

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='+') #go through all rows in first column (0) amd all rows in second column (1) 

k_range = range(1,10)
sse = []
for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age','Income($)']])
    sse.append(km.inertia_) #gives us SSE

plt.xlabel('x')
plt.ylabel('SSE')
plt.plot(k_range,sse)

plt.show()

