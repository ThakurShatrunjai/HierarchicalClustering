import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("DATA.csv.txt")
X=dataset.iloc[:, :].values

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean distance")
plt.show()

from sklearn.cluster import AgglomerativeClustering
clusterring= AgglomerativeClustering(n_clusters=5)
y_hc =clusterring.fit_predict(X)

plt.scatter(X[y_hc == 0,0],X[y_hc == 0,1], c='blue', label='Cluster_1')
plt.scatter(X[y_hc == 1,0],X[y_hc == 1,1], c='yellow', label='Cluster_2')
plt.scatter(X[y_hc == 2,0],X[y_hc == 2,1], c='green', label='Cluster_3')
plt.scatter(X[y_hc == 3,0],X[y_hc == 3,1], c='red', label='Cluster_4')
plt.scatter(X[y_hc == 4,0],X[y_hc == 4,1], c='brown', label='Cluster_5')
plt.title("Cluster Of Customers")
plt.xlabel("Annual Income $")
plt.ylabel("Spending Score [0-100]")
plt.legend()
plt.show()
