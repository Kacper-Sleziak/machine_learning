from collections import defaultdict

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def save_results(data_functions, classifiers):
    classifiers_results = defaultdict()

    for classifier in classifiers:
        classifier_results = defaultdict()

        model = classifier()
        classifier_name = model.__class__.__name__

        for data_function in data_functions:
            data = data_function[0]
            function_name = data_function[1]

            score = get_score(model, data)
            function_results = {
                "rkf": score,
            }

            classifier_results[function_name] = function_results
        classifiers_results[classifier_name] = classifier_results

    return classifiers_results


def get_score(model, data):
    X_train, X_test, y_train, y_test = data

    rkf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)
    score_skf = cross_val_score(model, X_test, y_test, cv=rkf)

    return score_skf


def print_means_and_deviations(classifiers_results):
    print("")
    cross_val_scores = {
        "gaussian_rkf": classifiers_results["GaussianNB"]["cancer"]["rkf"],
        "knearest_rkf": classifiers_results["KNeighborsClassifier"]["cancer"]["rkf"],
        "decision_tree_rkf": classifiers_results["DecisionTreeClassifier"]["cancer"]["rkf"],
    }

    print("---STANDARD DEVIATION---")
    for key, value in cross_val_scores.items():
        print(f"{key} - {value.std():.3f}")

    print("")

    print("---MEAN VALUE---")
    for key, value in cross_val_scores.items():
        print(f"{key} - {value.mean():.3f}")


data = load_breast_cancer()
classifiers = (KNeighborsClassifier, GaussianNB, DecisionTreeClassifier)
X = data.data
y = data.target

###
# PCA for training data only
###
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pca = PCA(0.8)
X_train = pca.fit_transform(X_train)

data = (X_train, X_test, y_train, y_test)
data_functions = ([data, "cancer"],)
classifiers_results = save_results(data_functions, classifiers)

print("PCA: TRAINING DATA PREPROCESSED")
print_means_and_deviations(classifiers_results)

print("\n\n")

###
# SelectKBest for training data only
###
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
select_k_best = SelectKBest(chi2, k=int(np.sqrt(X.shape[1])))
select_k_best.fit(X_train, y_train)
X_train = select_k_best.transform(X_train)

data = (X_train, X_test, y_train, y_test)
data_functions = ([data, "cancer"],)
classifiers_results = save_results(data_functions, classifiers)

print("SelectKBest: TRAINING DATA PREPROCESSED")
print_means_and_deviations(classifiers_results)
