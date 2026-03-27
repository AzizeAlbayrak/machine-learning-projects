import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('Mall_Customers.csv')
print(veriler.head())
print(veriler.shape)
print(veriler.describe())
print(veriler.dtypes)
print(veriler.isnull().sum())

#encoder
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
veriler["Gender"] = le.fit_transform(veriler["Gender"])
print(veriler)

veriler = veriler.drop('CustomerID' , axis = 1)

X = veriler.iloc[:,[2,3]].values


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)


from sklearn.cluster import KMeans

"""
kmeans = KMeans(n_clusters=3 , init = "k-means++")
kmeans.fit(X)

print(kmeans.cluster_centers_)
"""


sonuclar = []
for i in range(1,10):
    kmeans = KMeans(n_clusters=i , init="k-means++", random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)
    
plt.plot(range(1,10), sonuclar)
plt.show()

#Agglomerative ağaç yapısı, bailangıçta tek tek başlıyor, En yakınları yavaş yavaş birleştir
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=3)
Y_pred = ac.fit_predict(X)
print(Y_pred)

plt.scatter(X[Y_pred == 0,0] , X[Y_pred == 0,1], s =100, c= 'red')
plt.scatter(X[Y_pred == 1,0] , X[Y_pred == 1,1], s =100, c= 'blue')
plt.scatter(X[Y_pred == 2,0] , X[Y_pred == 2,1], s =100, c= 'green')
plt.show()

import scipy.cluster.hierarchy as sch
dendgrom =sch.dendrogram(sch.linkage(X, method='ward'))
plt.show()

kmeans = KMeans(n_clusters=3)
y_kmeans = kmeans.fit_predict(X)


from sklearn.metrics import silhouette_score

score = silhouette_score(X, Y_pred)
print("silhouette_score:" ,score)


from sklearn.metrics import davies_bouldin_score

db = davies_bouldin_score(X, Y_pred)
print("davies_bouldin_score:",db)

