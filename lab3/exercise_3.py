from sklearn.datasets import make_moons, make_circles, make_gaussian_quantiles
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from collections import defaultdict

from exercise_1 import CustomClassifier
from exercise_2 import CustomClassifier as CustomClassifier2
from icecream import ic


def save_results(data_functions, classifiers):
    """
    for every classifier:

    format_of_kf = {
        'kf2': [],
        'kf5': []
    }^

    format_of_results = {
        'make_circles': format_of_kf,
        'make_gaussian_quantiles': format_of_kf,
        'make_moons': format_of_kf
        }
    """

    classifiers_results = defaultdict()

    for classifier in classifiers:
        classifier_results = defaultdict()
        classifier_name = classifier.__name__

        for data_function in data_functions:
            function_name = data_function.__name__

            X, y = data_function()

            model = CustomClassifier()
            score_kf_2, score_kf_5 = get_scores(model, X, y)
            function_results = {
                "kf2": score_kf_2,
                "kf5": score_kf_5,
            }
            add_means_and_deviations(function_results)

            classifier_results[function_name] = function_results
        classifiers_results[classifier_name] = classifier_results

    return classifiers_results


def add_means_and_deviations(results):
    results["kf2_deviation"] = f"{results['kf2'].std():.3f}"
    results["kf5_deviation"] = f"{results['kf5'].std():.3f}"

    results["kf2_mean"] = f"{results['kf2'].mean():.3f}"
    results["kf5_mean"] = f"{results['kf5'].mean():.3f}"


def get_scores(model, X, y):
    kf_2 = KFold(n_splits=2)
    kf_5 = KFold(n_splits=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model.fit(X_train, y_train)

    score_kf_2 = cross_val_score(model, X_test, y_test, cv=kf_2)
    score_kf_5 = cross_val_score(model, X_test, y_test, cv=kf_5)
    return score_kf_2, score_kf_5


data_functions = (make_moons, make_circles, make_gaussian_quantiles)
classifiers = (CustomClassifier, CustomClassifier2)

classifiers_results = save_results(data_functions, classifiers)


ic(classifiers_results)
ic("")
ic(classifiers_results)
