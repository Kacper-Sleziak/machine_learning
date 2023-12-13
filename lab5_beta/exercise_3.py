import numpy as np
from scipy import stats

dataset = np.load('datasets.npy')
means = np.mean(dataset, axis=1)
ranks = stats.rankdata(means, axis=1)
classifiers = dataset.shape[2]
w_stat = np.empty(shape=(classifiers, classifiers))
p_val = np.empty(shape=(classifiers, classifiers))
results = np.empty(shape=(classifiers, classifiers), dtype='bool')
alpha = np.empty(shape=(classifiers, classifiers), dtype='bool')
cross = np.empty(shape=(classifiers, classifiers), dtype='bool')
classifiers_names = ["GNB","KNN","DT"]


for i in range(0, classifiers):
    for j in range(0, classifiers):
        w, p = stats.ranksums(ranks[:, i], ranks[:, j])
        w_stat[i, j] = w
        p_val[i, j] = p
        results[i, j] = w is not np.nan and w > 0
        alpha[i,j] = w is not np.nan and p < 0.05
        cross[i,j] = alpha[i,j] and results[i,j]

print(w_stat)
print(p_val)
print(results)
print(alpha)
print(cross)

def whichbetter(val1, val2, name1, name2) -> str :
    if val1 > val2:
        return ""+name1+" with rank "+str(val1)+" better than "+name2+" with rank "+str(val2)
    else:
        return ""+name2+" with rank "+str(val2)+" better than "+name1+" with rank "+str(val1)

print(whichbetter(ranks[0], ranks[1], classifiers_names[0], classifiers_names[1]))
print(whichbetter(ranks[0], ranks[2], classifiers_names[0], classifiers_names[2]))
print(whichbetter(ranks[2], ranks[1], classifiers_names[2], classifiers_names[1]))
