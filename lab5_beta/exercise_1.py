from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
from sklearn.base import clone

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)
classifiers = [GaussianNB(), KNeighborsClassifier(), DecisionTreeClassifier(random_state=8)]
results = np.empty(shape=(len(os.listdir('datasets')), 10, len(classifiers)))

for dataset_index, dataset_file in enumerate(os.listdir('datasets')):
    dataset = np.loadtxt('datasets/' + dataset_file, delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1]

    for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
        for classifier_id, classifier in enumerate(classifiers):
            clf = clone(classifier)
            clf.fit(X[train_index], y[train_index])
            predict_y = clf.predict(X[test_index])
            accuracy = accuracy_score(y[test_index], predict_y)
            results[dataset_index, i, classifier_id] = accuracy

# for j, classifier in enumerate(["Gaussian Naive Bayes", "k-Nearest", "Decision Tree"]):
#     print()
#     print("Result "+classifier+": "+str(round(np.mean(results[j]), 3))+"+/-"+str(round(np.std(results[j]), 3)))

np.save('datasets.npy', results)