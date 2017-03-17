import random
def cluster(Hr_vals):
	res = []
	for val in Hr_vals:
		res.append(random.randint(0,3))
	return res