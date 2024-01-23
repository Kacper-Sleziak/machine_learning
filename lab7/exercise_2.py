import numpy as np
from exercise_1 import SamplingClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from scipy import stats
from sklearn.datasets import make_classification
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB

X_arr, y_arr = np.empty([3], dtype=np.ndarray), np.empty([3], dtype=np.ndarray)

X_arr[0], y_arr[0] = make_classification(
    n_samples=5000, 
    n_features=4, 
    n_informative=2, 
    n_classes=2, 
    weights=[5 / 6, 1 / 6]
)

X_arr[1], y_arr[1] = make_classification(
    n_samples=5000, 
    n_features=4, 
    n_informative=2, 
    n_classes=2, 
    weights=[0.99, 0.01]
)

X_arr[2], y_arr[2] = make_classification(
    n_samples=5000,
    n_features=4,
    n_informative=2,
    n_classes=2,
    weights=[0.9, 0.1],
    flip_y=0.05,
)

preprocessors = [
    RandomOverSampler(random_state=10),
    RandomUnderSampler(random_state=11),
    SMOTE(random_state=12),
    None,
]


preoprocesors_len = len(preprocessors)
shape = (preoprocesors_len, preoprocesors_len, len(X_arr), 4)

results = np.zeros((preoprocesors_len, len(X_arr), 10, 4))
t_stat = np.empty(shape=shape)
p_val = np.empty(shape=shape)
results_ = np.empty(shape=shape, dtype="bool")
alpha = np.empty(shape=shape, dtype="bool")
cross = np.empty(shape=shape, dtype="bool")
means = np.empty(shape=shape)

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)

for dataset_id, (X, y) in enumerate(zip(X_arr, y_arr)):
    for preprocessor_id, preprocessor in enumerate(preprocessors):
        for fold_id, (train_index, test_index) in enumerate(rskf.split(X, y)):
            classifier = SamplingClassifier(
                base_estimator=GaussianNB(), base_preprocessing=preprocessor
            )
            
            classifier.fit(X[train_index], y[train_index])
            y_result = classifier.predict(X[test_index])

            results[preprocessor_id][dataset_id][fold_id][0] = f1_score(
                y[test_index], y_result, average="weighted"
            )
            
            results[preprocessor_id][dataset_id][fold_id][1] = balanced_accuracy_score(
                y[test_index], y_result
            )
            
            results[preprocessor_id][dataset_id][fold_id][2] = precision_score(
                y[test_index], y_result, average="weighted"
            )
            results[preprocessor_id][dataset_id][fold_id][3] = recall_score(
                y[test_index], y_result, average="weighted"
            )

for i in range(preoprocesors_len):
    for j in range(preoprocesors_len):
        for k in range(len(X_arr)):
            for m in range(4):
                t, p = stats.ttest_rel(results[i, k, :, m], results[j, k, :, m])
                
                t_stat[i, j, k, m] = t
                p_val[i, j, k, m] = p
                
                results_[i, j, k, m] = t is not np.nan and t > 0
                alpha[i, j, k, m] = t is not np.nan and p < 0.05
                cross[i, j, k, m] = alpha[i, j, k, m] and results_[i, j, k, m]
                
                means[i] = round(np.mean(results[i, k, :, m]), 3)


for i in range(len(X_arr)):
    for j in range(4):
        print()
        print(cross[i, j])
