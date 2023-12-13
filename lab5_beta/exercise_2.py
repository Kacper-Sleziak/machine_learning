import numpy as np
from scipy import stats

dataset = np.load('datasets.npy')
classifiers = dataset.shape[2]
t_stat = np.empty(shape=(classifiers, classifiers))
p_val = np.empty(shape=(classifiers, classifiers))
results = np.empty(shape=(classifiers, classifiers), dtype='bool')
alpha = np.empty(shape=(classifiers, classifiers), dtype='bool')
cross = np.empty(shape=(classifiers, classifiers), dtype='bool')
means = np.empty(shape=3)
classifiers_names = ["GNB","KNN","DT"]

dataset_id = 0

for i in range(0, classifiers):
    for j in range(0, classifiers):
        # t = okresla czy roznice sa istotne statysycznie
        # p informtuje o istotnosc istatystycznej
        t, p = stats.ttest_rel(dataset[dataset_id, :, i], dataset[dataset_id, :, j])
        t_stat[i, j] = t
        p_val[i, j] = p
        results[i, j] = t is not np.nan and t > 0
        alpha[i,j] = t is not np.nan and p < 0.05
        cross[i,j] = alpha[i,j] and results[i,j]
        means[i] = round(np.mean(dataset[dataset_id, :, i]), 3)

print(t_stat)
print(p_val)
print(results)
print(alpha)
print(cross)

def whichbetter(val1, val2, name1, name2) -> str :
    if val1 > val2:
        return ""+name1+" with "+str(val1)+" better than "+name2+" with "+str(val2)
    else:
        return ""+name2+" with "+str(val2)+" better than "+name1+" with "+str(val1)

print(whichbetter(means[0], means[1], classifiers_names[0], classifiers_names[1]))
print(whichbetter(means[0], means[2], classifiers_names[0], classifiers_names[2]))
print(whichbetter(means[2], means[1], classifiers_names[2], classifiers_names[1]))