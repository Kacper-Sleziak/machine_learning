from math import ceil, sqrt

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import check_X_y


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classifiers, base_classifier=GaussianNB()):
        self.n_classifiers = n_classifiers
        self.base_classifier = base_classifier
        self.classifiers = []

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        training_size = ceil(sqrt(len(X)))
        for _ in range(self.n_classifiers):
            indices = np.random.choice(len(X), training_size, replace=True)
            X_bootstrap, y_bootstrap = X[indices], y[indices]

            classifier = self.base_classifier.fit(X_bootstrap, y_bootstrap)
            self.classifiers.append(classifier)

    def predict(self, X):
        predictions = np.array([clf.predict(X) for clf in self.classifiers])

        final_predictions = self.majority_vote(predictions)
        return final_predictions

    def majority_vote(self, predictions):
        final_predictions = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions
        )

        return final_predictions


rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)
classifiers = [GaussianNB(), EnsembleClassifier(5)]
results = np.empty(shape=(10, len(classifiers)))

X, y = make_classification()
for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
    for classifier_id, classifier in enumerate(classifiers):
        clf = clone(classifier)
        clf.fit(X[train_index], y[train_index])

        predict_y = clf.predict(X[test_index])

        accuracy = accuracy_score(y[test_index], predict_y)
        results[i, classifier_id] = accuracy

for i, classifier in enumerate(["Gaussian Naive Bayes", "Ensemble"]):
    print(f"Result of {classifier}: {(round(np.mean(results[i]), 3))}")
