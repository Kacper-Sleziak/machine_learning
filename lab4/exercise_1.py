from collections import defaultdict

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, cross_val_score, train_test_split
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

            score_kf_2, score_kf_5 = get_scores(model, X, y)
            function_results = {
                "kf2": score_kf_2,
                "kf5": score_kf_5,
            }

            classifier_results[function_name] = function_results
        classifiers_results[classifier_name] = classifier_results

    return classifiers_results


def get_scores(model, X, y):
    kf_2 = KFold(n_splits=2)
    kf_5 = KFold(n_splits=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model.fit(X_train, y_train)

    score_kf_2 = cross_val_score(model, X_test, y_test, cv=kf_2)
    score_kf_5 = cross_val_score(model, X_test, y_test, cv=kf_5)
    return score_kf_2, score_kf_5


data = load_breast_cancer
data_functions = ([data(), "cancer"],)
classifiers = (KNeighborsClassifier, GaussianNB, DecisionTreeClassifier)
classifiers_results = save_results(data_functions, classifiers)

cross_val_scores = {
    "gaussian_kf2": classifiers_results["GaussianNB"]["cancer"]["kf2"],
    "gaussian_kf5": classifiers_results["GaussianNB"]["cancer"]["kf5"],
    "knearest_kf2": classifiers_results["KNeighborsClassifier"]["cancer"]["kf2"],
    "knearest_kf5": classifiers_results["KNeighborsClassifier"]["cancer"]["kf5"],
    "decision_tree_kf2": classifiers_results["DecisionTreeClassifier"]["cancer"]["kf2"],
    "decision_tree_kf5": classifiers_results["DecisionTreeClassifier"]["cancer"]["kf5"],
}

print("---STANDARD DEVIATION---")
for key, value in cross_val_scores.items():
    print(f"{key} - {value.std():.3f}")

print("")

print("---MEAN VALUE---")
for key, value in cross_val_scores.items():
    print(f"{key} - {value.mean():.3f}")
