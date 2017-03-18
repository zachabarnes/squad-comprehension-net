from sklearn.cluster import KMeans
import numpy as np

X = np.load('/mnt/encoded_files/encoded0.npz')['data']
for i in xrange(1,3):
	print i
	filen = '/mnt/encoded_files/encoded' + str(i) + '.npz'
	X = np.concatenate((X,np.load(filen)['data']),axis=0)

kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

print kmeans.labels_
print kmeans.cluster_centers_
