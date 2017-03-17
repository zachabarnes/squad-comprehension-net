from sklearn.cluster import  KMeans
import numpy as np

X = np.load('data/autoencoded.npz')
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

print kmeans.labels_
print kmeans.predict([[0, 0], [4, 4]])
print kmeans.cluster_centers_
