import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier


class CustomeClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        
        return self
    
    def predict(self, X):
        distances = cdist(X, self.X_)
        distances = np.argsort(distances, axis=1)

        k_nearest_distances = distances[:, :self.k]
        k_nearest_labels = self.y_[k_nearest_distances]
        
        predictions = np.zeros(len(X))
        
        for i in range(X.shape[0]):
            unique, counts = np.unique(k_nearest_labels[i], return_counts=True)
            
            predicted_label = unique[np.argmax(counts)]
            predictions[i] = predicted_label

        return predictions


k = 5          
X, y = make_classification(500)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = CustomeClassifier(k)
model.fit(X_train, y_train)
predicted_labels = model.predict(X_test)
custom_model_accuracy = accuracy_score(y_test, predicted_labels)

model= KNeighborsClassifier(k)
model.fit(X_train, y_train)
predicted_labels = model.predict(X_test)
kneighbors_model_accuracy = accuracy_score(y_test, predicted_labels)

print(custom_model_accuracy == kneighbors_model_accuracy)