import random

def read_csv(file: str)->tuple:
	list_of_data = []
	with open(file) as f:
		attributes = list(f.readline().split(','))
		while True:
			curr_row = f.readline()
			if len(curr_row) == 0:
				break
			ps_vals = list(map(float, curr_row.split(',')))
			list_of_data.append(ps_vals)
	return (attributes, list_of_data)

def normalize_data(list_of_data: list):
	data_size = len(list_of_data)
	for attr_ind in range(len(list_of_data[0])):
		max_val = 0
		min_val = list_of_data[0][attr_ind]
		for row in range(data_size):
			max_val = max(list_of_data[row][attr_ind], max_val)
			min_val = min(list_of_data[row][attr_ind], min_val)

		for row in range(data_size):
			list_of_data[row][attr_ind] /= (max_val - min_val)

def get_dist(point_1: list, point_2: list) -> float:
	curr_dis = 0
	for attr_ind in range(len(point_1)):
			curr_dis += ((point_1[attr_ind] - point_2[attr_ind])**2)
	return curr_dis**0.5

# returns the index of closest center
def closest_centre(centers: list, data: list)->int:
	closest_ind = -1
	min_dis = -1
	for ind in range(len(centers)):
		curr_dis = get_dist(centers[ind], data)
		if min_dis == -1 or curr_dis <= min_dis:
			closest_ind = ind
			min_dis = curr_dis
	return closest_ind

def get_curr_cluster(centers: list, list_of_data: list)->list:
	k = len(centers)
	clusters = [[] for _ in range(k)]
	for curr_data in list_of_data:
		clusters[closest_centre(centers, curr_data)].append(curr_data)
	return clusters

def get_center(cluster: list):
	size_of_cluster = len(cluster)
	dim = (len(cluster[0]) - 1)
	center = [0 for _ in range(dim)]
	for curr_data in cluster:
		# assert(len(curr_data) == 9)
		for attr_ind in range(dim):
			center[attr_ind] += curr_data[attr_ind]
	for attr_ind in range(dim):
		center[attr_ind] /= size_of_cluster
	return center

def get_centers(clusters: list):
	centers = []
	for cluster in clusters:
		if len(cluster) == 0:
			return False
		centers.append(get_center(cluster))
	return centers

def heuristic_based_init(k, full_data):
	indices = [i for i in range(len(full_data))]
	c0 = random.choice(indices)
	centers = {c0}

	for centr_id in range(k - 1):
		max_dist = 0
		ps_center = -1
		for data_id in range(len(full_data)):
			if data_id in centers:
				continue
			min_dis = get_dist(full_data[c0][:-1], full_data[data_id][:-1])
			for prev_id in centers:
				min_dis = min(min_dis, get_dist(full_data[prev_id][:-1], full_data[data_id][:-1]))
			if min_dis >= max_dist:
				max_dist = min_dis
				ps_center = data_id
		centers.add(ps_center)
	return centers

def random_center_init(k, full_data):		
	indices = [i for i in range(len(full_data))]
	chosen_centers = set(random.sample(indices, k))
	return chosen_centers

# give the optional value of tolerance if 
# convergence is taking too much time, by
# default it is zero
def k_means(k, init_func, list_of_data: list, tol = 0, get_stats=0)->list:
	size_data = len(list_of_data)
	assert(k <= size_data)
	center_ind = init_func(k, list_of_data)
	centers = [list_of_data[ind] for ind in center_ind]
	it = 0
	while True:
		it += 1
		clusters = get_curr_cluster(centers, list_of_data)
		if True in [len(cluster) == 0 for cluster in clusters]:
			center_ind = init_func(k, list_of_data)
			centers = [list_of_data[ind] for ind in center_ind]
			if get_stats:
				print("empty cluster came, restarting with different initialization")
			it = 0
			continue
		else:
			new_centers = get_centers(clusters)		
		mx_dist = 0
		for ind in range(len(centers)):
			mx_dist = max(mx_dist, get_dist(new_centers[ind], centers[ind]))
		if(get_stats):
			print("Iteration %d done" %it)
		if mx_dist <= tol:
			break
		centers = [centre for centre in new_centers]
	clusters = get_curr_cluster(centers, list_of_data)
	if(get_stats):
		print("The number of iterations required for convergence:", it)
	return clusters

def get_pred_for_cluster(cluster: list):
	pred = 0
	lab_counts = {0: 0, 1: 0}
	for curr_data in cluster:
		lab_counts[int(curr_data[-1])] += 1
	if lab_counts[0] > lab_counts[1]:
		return 0
	return 1

def get_variance_of_cluster(cluster: list):
	center = get_center(cluster)
	var_clust = 0
	for data in cluster:
		ps_dis = get_dist(center, data)
		var_clust += (ps_dis ** 2)
	return var_clust

def get_total_variance(clusters: list):
	tot_var = 0
	for cluster in clusters:
		tot_var += get_variance_of_cluster(cluster)
	return tot_var

def get_predictions_for_clusters(clusters: list):
	y_pred = []
	for cluster in clusters:
		pred = get_pred_for_cluster(cluster)
		ps_siz = len(cluster)
		y_pred.extend([pred for _ in range(ps_siz)])
	return y_pred

def get_clusters_and_preds_for_test(test_data: list, clusters: list)->tuple:
	y_pred = []
	centers = get_centers(clusters)
	predictions_for_clusters = [get_pred_for_cluster(clusters[i]) for i in range(len(clusters))]
	clusters_test_ind = []
	for data in test_data:
		clust_ind = closest_centre(centers, data)
		y_pred.append(predictions_for_clusters[clust_ind])
		clusters_test_ind.append(clust_ind)
	return (clusters_test_ind, y_pred)