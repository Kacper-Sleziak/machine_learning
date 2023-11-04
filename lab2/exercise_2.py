from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from numpy import argmax
from sklearn.metrics import accuracy_score
from matplotlib.pyplot  import subplots, scatter, title, show, savefig

X, y = make_classification(400, n_informative=2, n_redundant=0, n_repeated=0, flip_y=0.08)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = GaussianNB()
model.fit(X_train, y_train)

predicted_labels = model.predict_proba(X_test)
predicted_labels = argmax(predicted_labels, axis=1)

score = accuracy_score(y_test, predicted_labels)
print(score)

fig, (ax1, ax2) = subplots(1, 2)

first_attr = X[:, 0]
second_attr = X[:, 1]
ax1.scatter(first_attr, second_attr, c=y)

first_attr = X_test[:, 0]
second_attr = X_test[:, 1]
ax2.scatter(first_attr, second_attr, c=predicted_labels)

fig.suptitle(f"Score {score:.3f}")
savefig('erxercise_2_fig')