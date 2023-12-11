from collections import defaultdict

from sklearn.datasets import load_breast_cancer
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
            X = data_function[0].data
            y = data_function[0].target
            function_name = data_function[1]

            score = get_score(model, X, y)
            function_results = {
                "rkf": score,
            }

            classifier_results[function_name] = function_results
        classifiers_results[classifier_name] = classifier_results

    return classifiers_results


def get_score(model, X, y):
    rkf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)
    score_skf = cross_val_score(model, X, y, cv=rkf)

    return score_skf


data = load_breast_cancer
data_functions = ([data(), "cancer"],)
classifiers = (KNeighborsClassifier, GaussianNB, DecisionTreeClassifier)
classifiers_results = save_results(data_functions, classifiers)

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
