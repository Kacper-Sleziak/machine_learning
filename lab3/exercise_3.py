from sklearn.datasets import make_moons, make_circles, make_gaussian_quantiles
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from numpy import std

from exercise_1 import CustomClassifier
from exercise_2 import CustomClassifier as CustomClassifier2
from icecream import ic


def get_results_for_given_model(model, X, y):
    kf_2 = KFold(n_splits=2)
    kf_5 = KFold(n_splits=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model.fit(X_train, y_train)

    score_kf_2 = cross_val_score(model, X_test, y_test, cv=kf_2)
    score_kf_5 = cross_val_score(model, X_test, y_test, cv=kf_5)
    return score_kf_2, score_kf_5


def save_results(data_functions):
    """
    format_of_kf = {
        'kf2': [],
        'kf5': []
    }

    format_of_results = {
        'make_circles': format_of_kf,
        'make_gaussian_quantiles': format_of_kf,
        'make_moons': format_of_kf
        }
    """

    custom_classifier_results = {}
    custom_classifier2_results = {}

    for data_function in data_functions:
        function_name = data_function.__name__

        X, y = data_function()

        model = CustomClassifier()
        score_kf_2, score_kf_5 = get_results_for_given_model(model, X, y)
        custom_classifier_results[function_name] = {
            'kf2': score_kf_2,
            'kf5': score_kf_5,
        }

        model = CustomClassifier2(k=5)
        score_kf_2, score_kf_5 = get_results_for_given_model(model, X, y)
        custom_classifier2_results[function_name] = {
            'kf2': score_kf_2,
            'kf5': score_kf_5,
        }

    return custom_classifier_results, custom_classifier2_results


def add_deviations(classifier_results):
    for function in classifier_results:
        classifier_results[function]['kf2_deviation'] = f"{classifier_results[function]['kf2'].std():.3f}"
        classifier_results[function]['kf5_deviation'] = f"{classifier_results[function]['kf5'].std():.3f}"


def add_means(classifier_results):
    for function in classifier_results:
        classifier_results[function]['kf2_mean'] = f"{classifier_results[function]['kf2'].mean():.3f}"
        classifier_results[function]['kf5_mean'] = f"{classifier_results[function]['kf5'].mean():.3f}"


data_functions = [make_moons, make_circles, make_gaussian_quantiles]
custom_classifier_results, custom_classifier2_results = save_results(data_functions)

add_deviations(custom_classifier_results)
add_deviations(custom_classifier2_results)

add_means(custom_classifier_results)
add_means(custom_classifier2_results)

ic(custom_classifier_results)
ic("")
ic(custom_classifier2_results)


