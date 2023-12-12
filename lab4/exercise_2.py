from collections import defaultdict

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
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
            score2 = get_score(model, data, True)
            function_results = {
                "rkf_scaled_one": score,
                "rkf_both_scaled": score2,
            }

            classifier_results[function_name] = function_results
        classifiers_results[classifier_name] = classifier_results

    return classifiers_results


def get_score(model, data, scale_both=False):
    X, y = data
    rkf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)
    scaler = StandardScaler()

    scores = []
    for i, (train_index, test_index) in enumerate(rkf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = scaler.fit_transform(X_train)
        if scale_both:
            X_test = scaler.transform(X_test)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = (y_pred == y_test).mean()
        scores.append(accuracy)

    score_skf = cross_val_score(model, X, y, cv=rkf)

    return score_skf


def print_means_and_deviations(classifiers_results, choice="rkf_scaled_one"):
    print("")
    cross_val_scores_scaled_one = {
        "gaussian_rkf": classifiers_results["GaussianNB"]["cancer"][choice],
        "knearest_rkf": classifiers_results["KNeighborsClassifier"]["cancer"][choice],
        "decision_tree_rkf": classifiers_results["DecisionTreeClassifier"]["cancer"][
            choice
        ],
    }

    print("---STANDARD DEVIATION---")
    for key, value in cross_val_scores_scaled_one.items():
        print(f"{key} - {value.std():.3f}")

    print("")

    print("---MEAN VALUE---")
    for key, value in cross_val_scores_scaled_one.items():
        print(f"{key} - {value.mean():.3f}")


data = load_breast_cancer()
classifiers = (KNeighborsClassifier, GaussianNB, DecisionTreeClassifier)
X = data.data
y = data.target

###
# Standard Scaler for training data only
###
data_functions = ([(X, y), "cancer"],)
classifiers_results = save_results(data_functions, classifiers)

print("TRAINING DATA SCALED")
print_means_and_deviations(classifiers_results)

print("\n\n")

###
# Standard Scaler for training and test data
###
print("TRAINING AND TEST DATA SCALED")
print_means_and_deviations(classifiers_results, choice="rkf_both_scaled")
