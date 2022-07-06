import k_means
from sklearn import metrics

attributes, list_of_data = k_means.read_csv('mean_data.csv')
k_means.normalize_data(list_of_data)
print("Enter the value of k:", end=' ')
k = int(input())

def get_metrics(y_label, y_pred):
	# metrics with ground truth
	print("\nThe metrics with ground truth are:")
	print("The homogeneity score is:", metrics.homogeneity_score(y_label, y_pred))
	print("NMI is:", metrics.normalized_mutual_info_score(y_label, y_pred))
	print("ARI is:", metrics.adjusted_rand_score(y_label, y_pred))
	print("The accuracy is:", metrics.accuracy_score(y_label, y_pred))

	# metrics without ground truth
	print("\nThe metrics without ground truth are:")
	print("The silhouette score is:", metrics.silhouette_score(X, y_pred))
	print("The calinski harabasz score is:", metrics.calinski_harabasz_score(X, y_pred))


clusters = k_means.k_means(k, k_means.random_center_init, list_of_data)
y_pred = k_means.get_predictions_for_clusters(clusters)

X, y_label = [], []
for cluster in clusters:
	for data in cluster:
		X.append(data[:-1])
		y_label.append(int(data[-1]))

get_metrics(y_label, y_pred)