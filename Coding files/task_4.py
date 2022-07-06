import k_means, random
from sklearn import metrics

attributes, list_of_data = k_means.read_csv('mean_data.csv')
k_means.normalize_data(list_of_data)

# get the value from part iii)
k = 10

def get_stats(data: list):
	mean_data = 0
	mean_data2 = 0
	for val in data:
		mean_data += val
		mean_data2 += val**2
	mean_data2 /= len(data)
	mean_data /= len(data)
	var_data = mean_data2 - mean_data**2
	return var_data, mean_data

def evaluate_k_means_with_k_centers(full_data, init_method)->dict:
	indices = []
	centers_ind = init_method(k, full_data)
	init_centers = []

	for i in range(len(full_data)):
		if i in centers_ind:
			init_centers.append(full_data[i][:-1])
		else:
			indices.append(i)

	# metrics
	NMI_sc, ARI_sc, Hom_sc, acc_sc = 0, 0, 0, 0

	size = len(list_of_data)
	test_size = int(0.2*(size - k))
	
	for _ in range(50):
		train_data, test_data = [], []
		indices_test_points = random.sample(indices, test_size)

		for i in range(size):
			if i in indices_test_points:
				test_data.append(list_of_data[i])
			else:
				train_data.append(list_of_data[i])

		clusters = k_means.k_means(k, init_method, train_data)
		clust_test_ind, y_pred = k_means.get_clusters_and_preds_for_test(test_data, clusters)
		y_label = [test_data[i][-1] for i in range(test_size)]

		NMI_sc += metrics.normalized_mutual_info_score(y_label, y_pred)
		ARI_sc += metrics.adjusted_rand_score(y_label, y_pred)
		Hom_sc += metrics.homogeneity_score(y_label, y_pred)
		acc_sc += metrics.accuracy_score(y_label, y_pred)

	NMI_sc /= 50
	ARI_sc /= 50
	Hom_sc /= 50
	acc_sc /= 50

	return [NMI_sc, ARI_sc, Hom_sc]

def evaluate_init(k: int, full_data: list, method, method_name)->None:
	print("Testing", method_name)
	metrics = [[],[],[]]
	for r_i in range(50):
		ps_metrics = evaluate_k_means_with_k_centers(full_data, method)
		for i in range(3):
			metrics[i].append(ps_metrics[i])
		print("got metrics for run =", r_i + 1)
	
	import matplotlib.pyplot as plt
	figure, ax = plt.subplots()
	colors = ['blue', 'red', 'green']

	for i in range(3):
		ax.plot(metrics[i], color=colors[i])
	print("\nWith", method_name + ", the variance of metrics were:")
	st_nmi = get_stats(metrics[0])
	print("The variance of NMI:", st_nmi[0], "and the mean of NMI:", st_nmi[1])
	st_ari = get_stats(metrics[1])
	print("The variance of ARI:", st_ari[0], "and the mean of ARI:", st_ari[1])
	st_hom = get_stats(metrics[2])
	print("The variance of homogeneity:", st_hom[0], "and the mean of homogeneity:", st_hom[1])
	plt.savefig(method_name+'.png')
	plt.close()

evaluate_init(k, list_of_data, k_means.random_center_init, "random_center_init")
evaluate_init(k, list_of_data, k_means.heuristic_based_init, "heuritics_based_init")








