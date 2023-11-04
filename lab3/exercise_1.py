import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class CustomeClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):

        self.classes_size = len(np.unique(y))
        return self
    
    def predict(self, X):
        return np.random.choice(self.classes_size, len(X))
    
model = CustomeClassifier()

X, y = make_classification(500)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)
predicted_labels = model.predict(X_test)

print(accuracy_score(y_test, predicted_labels))