from sklearn.cluster import  KMeans
import numpy as np
from autoencoder import autoencoder


X = np.load('data/autoencoded.npz')
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

print kmeans.labels_
print kmeans.cluster_centers_
