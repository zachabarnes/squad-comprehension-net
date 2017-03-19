import numpy as np
from sklearn.metrics import mean_squared_error


def cluster(Hr_vals):
	clusters = np.load('data/kmeans_labels.npz')['data']
	res = []
	for val in Hr_vals:
		min_cost = None
		min_cluster = None
		for i in xrange(0,len(clusters)):
			d = mean_squared_error(val,clusters[i])
			if min_cost == None:
				min_cost = d
				min_cluster = i
			else:
				if d < min_cost:
					min_cost = d
					min_cluster = i
		result.append(min_cluster)


	return res

