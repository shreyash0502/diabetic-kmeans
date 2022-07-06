import k_means

attributes, list_of_data = k_means.read_csv('mean_data.csv')
k_means.normalize_data(list_of_data)
print("Enter the value of k: ", end=' ')
k = int(input())

#without tolerance(0 by default)
clusters = k_means.k_means(k, k_means.random_center_init, list_of_data, get_stats=1)

#with tolerance of 0.005
clusters = k_means.k_means(k, k_means.random_center_init, list_of_data, tol=0.005, get_stats=1)