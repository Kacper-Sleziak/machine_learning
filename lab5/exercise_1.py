import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# data = np.loadtxt('sciezka_do_pliku.txt', delimiter=',')
# X = data[:, :-1]  # Wszystkie kolumny oprócz ostatniej to cechy
# y = data[:, -1]   # Ostatnia kolumna to etykiety

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

classifiers = [GaussianNB(), KNeighborsClassifier(), DecisionTreeClassifier()]

n_splits = 2
n_repeats = 5
rkf = RepeatedKFold(n_splits=n_splits, n_repeats=5)
results_array = np.zeros((len(classifiers), n_splits, n_repeats))

for i, clf in enumerate(classifiers):
    # Iterate over repetitions
    for j, (train_index, test_index) in enumerate(rkf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit and predict
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Calculate accuracy and store in results_array
        accuracy = np.mean(y_pred == y_test)
        results_array[i, j, :] = accuracy

np.save('results_array.npy', results_array)
