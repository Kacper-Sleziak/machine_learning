from math import ceil, sqrt

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import check_X_y


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classifiers, soft, weighted, base_classifier=GaussianNB()):
        self.weighted = weighted
        self.soft = soft
        self.n_classifiers = n_classifiers
        self.base_classifier = base_classifier
        self.classifiers = []
        self.weighted_pred = []

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        training_size = ceil(sqrt(len(X)))
        for _ in range(self.n_classifiers):
            pool = np.random.choice(len(X), training_size, replace=True)
            X_bootstrap, y_bootstrap = X[pool], y[pool]

            classifier = self.base_classifier.fit(X_bootstrap, y_bootstrap)
            self.classifiers.append(classifier)

            prediction = clf.predict(X)
            self.weighted_pred.append(accuracy_score(y, prediction))

    def predict(self, X):
        if not self.soft and self.weighted:
            raise Exception("Weighted flag only works with soft voting!")

        if self.soft:
            final_predictions = self.vote_hard(X)
        else:
            final_predictions = self.vote_soft(X)
        return final_predictions

    def vote_hard(self, X):
        predictions = np.array([clf.predict(X) for clf in self.classifiers])
        final_predictions = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions
        )

        return final_predictions

    def vote_soft(self, X):
        classifiers_results = np.zeros(shape=(X.shape[0], len(classifiers)))

        if self.weighted:
            for i, classifier in enumerate(self.classifiers):
                classifiers_results += (
                    classifier.predict_proba(X) * self.weighted_pred[i]
                )
        else:
            for classifier in self.classifiers:
                classifiers_results += classifier.predict_proba(X)
        final_predictions = np.argmax(classifiers_results, axis=1)

        return final_predictions


rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)
classifiers = [EnsembleClassifier(5, True, True), EnsembleClassifier(5, True, False)]
results = np.empty(shape=(10, len(classifiers)))

X, y = make_classification()

for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
    for classifier_id, classifier in enumerate(classifiers):
        clf = clone(classifier)
        clf.fit(X[train_index], y[train_index])

        predict_y = clf.predict(X[test_index])

        accuracy = accuracy_score(y[test_index], predict_y)
        results[i, classifier_id] = accuracy

for i, classifier in enumerate(["Weighted soft", "Not weighted soft"]):
    print(f"Result of {classifier}: {(round(np.mean(results[i]), 3))}")
