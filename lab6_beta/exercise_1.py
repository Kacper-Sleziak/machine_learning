from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import numpy as np
from math import ceil, sqrt


class Ensemble(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classifiers):
        super()
        self.n_classifiers = n_classifiers
        self.classifiers = []
        for i in range(self.n_classifiers):
            self.classifiers.append(GaussianNB())

    def fit(self, X, y):
        training_size = ceil(sqrt(len(X)))
        for classifier in self.classifiers:
            pool = np.random.choice(np.arange(X.shape[0]), training_size, replace=True)
            classifier.fit(X[pool], y[pool])

    def predict(self, X):
        classifiers_results = []
        for classifier in self.classifiers:
            classifiers_results.append(classifier.predict(X))
        results = np.array(classifiers_results).T
        end_result = []
        for row in results:
            end_result.append(np.argmax(np.bincount(row)))
        return end_result


rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)
classifiers = [GaussianNB(), Ensemble(7)]

X, y = make_classification()
results = np.empty(shape=(10, len(classifiers)))
for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
    for classifier_id, classifier in enumerate(classifiers):
        clf = clone(classifier)
        clf.fit(X[train_index], y[train_index])
        predict_y = clf.predict(X[test_index])
        accuracy = accuracy_score(y[test_index], predict_y)
        results[i, classifier_id] = accuracy

for j, classifier in enumerate(["Gaussian Naive Bayes", "Ensemble"]):
    print()
    print("Result " + classifier + ": " + str(round(np.mean(results[j]), 3)) + "+/-" + str(round(np.std(results[j]), 3))