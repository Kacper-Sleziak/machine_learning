from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from numpy import argmax, array
from sklearn.preprocessing import StandardScaler

X, y = make_classification(400, n_informative=2, n_redundant=0, n_repeated=0, flip_y=0.08)
rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2)
scores = []

for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = GaussianNB()
    model.fit(X_train, y_train)

    predicted_labels = model.predict_proba(X_test)
    predicted_labels = argmax(predicted_labels, axis=1)

    score = accuracy_score(y_test, predicted_labels)
    scores.append(score)

scores = array(scores)

mean = scores.mean()
std = scores.std()
