from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

def save_results(data_functions, classifiers):
    classifiers_results = defaultdict()

    for classifier in classifiers:
        classifier_results = defaultdict()

        model = classifier()
        classifier_name = model.__class__.__name__

        for data_function in data_functions:
            data = data_function[0]
            function_name = data_function[1]

            score_kf_2, score_kf_5 = get_scores(model, data)
            function_results = {
                "kf2": score_kf_2,
                "kf5": score_kf_5,
            }

            classifier_results[function_name] = function_results
        classifiers_results[classifier_name] = classifier_results

    return classifiers_results


def get_scores(model, data):
    kf_2 = KFold(n_splits=2)
    kf_5 = KFold(n_splits=5)

    X_train, X_test, y_train, y_test = data

    # Added scaler
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)

    model.fit(X_train, y_train)

    score_kf_2 = cross_val_score(model, X_test, y_test, cv=kf_2)
    score_kf_5 = cross_val_score(model, X_test, y_test, cv=kf_5)
    return score_kf_2, score_kf_5


def print_means_and_deviations(classifiers_results):
    print("")
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

data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

# Standard Scaler for training data only
data = (X_train, X_test, y_train, y_test)
data_functions = ([data, "cancer"],)
classifiers = (KNeighborsClassifier, GaussianNB, DecisionTreeClassifier)
classifiers_results = save_results(data_functions, classifiers)

print("TRAINING DATA SCALED")
print_means_and_deviations(classifiers_results)

print("\n\n")

# Standard Scaler for training and test data
X_test = scaler.transform(X_test)
data = (X_train, X_test, y_train, y_test)
data_functions = ([data, "cancer"],)
classifiers = (KNeighborsClassifier, GaussianNB, DecisionTreeClassifier)

classifiers_results = save_results(data_functions, classifiers)
print("TRAINING AND TEST DATA SCALED")
print_means_and_deviations(classifiers_results)

