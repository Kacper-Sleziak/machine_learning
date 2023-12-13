import numpy as np
from scipy import stats

dataset = np.load("datasets.npy")
classifiers = dataset.shape[2]

t_statistic = np.empty(shape=(classifiers, classifiers))
p_value = np.empty(shape=(classifiers, classifiers))
results = np.empty(shape=(classifiers, classifiers), dtype="bool")
alpha = np.empty(shape=(classifiers, classifiers), dtype="bool")
cross = np.empty(shape=(classifiers, classifiers), dtype="bool")
means = np.empty(shape=3)

classifiers_names = ["GNB", "KNN", "DT"]
clf_len = len(classifiers)
dataset_id = 0

for i in range(clf_len):
    for j in range(clf_len):
        if i != j:
            t, p = stats.ttest_rel(dataset[dataset_id, :, i], dataset[dataset_id, :, j])

            t_statistic[i, j] = t
            p_value[i, j] = p
            results[i, j] = not np.isnan(t) and t > 0
            alpha[i, j] = not np.isnan(t) and p < 0.05
            cross[i, j] = alpha[i, j] and results[i, j]
            means[i] = round(np.mean(dataset[dataset_id, :, i]), 3)


def find_better_pair(val1, val2, name1, name2) -> str:
    if val1 > val2:
        return f"{name1} with {val1} better than {name2} with {val2}"
    else:
        return f"{name2} with {val2} better than {name1} with {val1}"


print(t_statistic)
print(p_value)
print(results)
print(alpha)
print(cross)

print(find_better_pair(means[0], means[1], classifiers_names[0], classifiers_names[1]))
print(find_better_pair(means[0], means[2], classifiers_names[0], classifiers_names[2]))
print(find_better_pair(means[2], means[1], classifiers_names[2], classifiers_names[1]))
