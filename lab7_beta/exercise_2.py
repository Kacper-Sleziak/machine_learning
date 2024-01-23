from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score, precision_score, balanced_accuracy_score, f1_score 
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from scipy import stats
from exercise_1 import SamplingClassifier

Xs, ys = np.empty([3], dtype=np.ndarray), np.empty([3], dtype=np.ndarray)
Xs[0], ys[0] = make_classification(n_samples=5000, n_features=4, n_informative=2, n_classes=2, weights=[5/6,1/6])
Xs[1], ys[1] = make_classification(n_samples=5000, n_features=4, n_informative=2, n_classes=2, weights=[0.99,0.01])
Xs[2], ys[2] = make_classification(n_samples=5000, n_features=4, n_informative=2, n_classes=2, weights=[0.9,0.1], flip_y=0.05)

preprocessors_names = ["RandomOverSampler", "RandomUnderSampler", "SMOTE", "None"]
preprocessors = [RandomOverSampler(random_state=10), RandomUnderSampler(random_state=11), SMOTE(random_state=12), None]

results = np.zeros((len(preprocessors), len(Xs), 10, 4))
t_stat = np.empty(shape=(len(preprocessors), len(preprocessors), len(Xs), 4))
p_val = np.empty(shape=(len(preprocessors), len(preprocessors), len(Xs), 4))
results_ = np.empty(shape=(len(preprocessors), len(preprocessors), len(Xs), 4), dtype='bool')
alpha = np.empty(shape=(len(preprocessors), len(preprocessors), len(Xs), 4), dtype='bool')
cross = np.empty(shape=(len(preprocessors), len(preprocessors), len(Xs), 4), dtype='bool')
means= np.empty(shape=(len(preprocessors), len(Xs), 4))

rskf = RepeatedStratifiedKFold(n_splits = 2, n_repeats = 5)
for dataset_id, (X, y) in enumerate(zip(Xs, ys)):
    for preprocessor_id, preprocessor in enumerate(preprocessors):
        for fold_id, (train_index, test_index) in enumerate(rskf.split(X, y)):
            classifier = SamplingClassifier(base_estimator=GaussianNB(), base_preprocessing=preprocessor)
            classifier.fit(X[train_index], y[train_index])
            y_result = classifier.predict(X[test_index])
            results[preprocessor_id][dataset_id][fold_id][0] = f1_score(y[test_index], y_result, average='weighted')
            results[preprocessor_id][dataset_id][fold_id][1] = balanced_accuracy_score(y[test_index], y_result)
            results[preprocessor_id][dataset_id][fold_id][2] = precision_score(y[test_index], y_result, average='weighted')
            results[preprocessor_id][dataset_id][fold_id][3] = recall_score(y[test_index], y_result, average='weighted')


for i in range(len(preprocessors)):
    for j in range(len(preprocessors)):
        for k in range(len(Xs)):
            for m in range(4):
                t, p = stats.ttest_rel(results[i,k, :, m], results[j,k, :, m])
                t_stat[i, j, k, m] = t
                p_val[i, j, k, m]  = p
                results_[i, j, k, m]  = t is not np.nan and t > 0
                alpha[i, j, k, m]  = t is not np.nan and p < 0.05
                cross[i, j, k, m] = alpha[i, j, k, m]  and results_[i, j, k, m] 
                means[i] = round(np.mean(results[i, k, :, m]), 3)


for i in range(len(Xs)):
    for j in range(4):
        print()
        print(cross[i,j])