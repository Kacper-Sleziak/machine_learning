from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


class SamplingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, base_preprocessing=None):
        super()
        self.base_estimator = base_estimator
        self.base_preprocessing = base_preprocessing

    def fit(self, X, y):
        X_inner = X
        y_inner = y

        if self.base_preprocessing:
            X_inner, y_inner = self.base_preprocessing.fit_resample(X, y)

        self.base_estimator.fit(X_inner, y_inner)
        return self

    def predict(self, X):
        return self.base_estimator.predict(X)


X, y = make_classification(random_state=12)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

classifier = SamplingClassifier(base_estimator=GaussianNB())
classifier.fit(X_train, y_train)
y_result = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_result)
