import numpy as np
from scipy import stats

dataset = np.load("datasets.npy")
classifiers = dataset.shape[2]
means = np.mean(dataset, axis=1)
ranks = stats.rankdata(means, axis=1)

w_statistic = np.empty(shape=(classifiers, classifiers))
p_value = np.empty(shape=(classifiers, classifiers))
results = np.empty(shape=(classifiers, classifiers), dtype="bool")
alpha = np.empty(shape=(classifiers, classifiers), dtype="bool")
cross = np.empty(shape=(classifiers, classifiers), dtype="bool")

classifiers_names = ["GNB", "KNN", "DT"]
clf_len = len(classifiers)

for i in range(clf_len):
    for j in range(clf_len):
        if i != j:
            w, p = stats.ranksums(ranks[:, i], ranks[:, j])

            w_statistic[i, j] = w
            p_value[i, j] = p
            results[i, j] = not np.isnan(w) and w > 0
            alpha[i, j] = not np.isnan(w) and p < 0.05
            cross[i, j] = alpha[i, j] and results[i, j]


print(w_statistic)
print(p_value)
print(results)
print(alpha)
print(cross)


def find_better_pair(val1, val2, name1, name2) -> str:
    if val1 > val2:
        return f"{name1} with {val1} better than {name2} with {val2}"
    return f"{name2} with {val2} better than {name1} with {val1}"


print(find_better_pair(ranks[0], ranks[1], classifiers_names[0], classifiers_names[1]))
print(find_better_pair(ranks[0], ranks[2], classifiers_names[0], classifiers_names[2]))
print(find_better_pair(ranks[2], ranks[1], classifiers_names[2], classifiers_names[1]))
