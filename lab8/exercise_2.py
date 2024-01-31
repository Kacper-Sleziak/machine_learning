import matplotlib.pyplot as plt
from scipy import stats
from sklearn.base import ClassifierMixin, clone
from sklearn.naive_bayes import GaussianNB
from strlearn.evaluators import TestThenTrain
from strlearn.metrics import balanced_accuracy_score
from strlearn.streams import StreamGenerator


class EnsembleClassifier(ClassifierMixin):
    def __init__(self, base_classifier, n_classifiers):
        self.base_classifier = base_classifier
        self.n_classifiers = n_classifiers
        self.pool = []

    def partial_fit(self, X, y, classes):
        new_classifier = clone(self.base_classifier)
        new_classifier.fit(X, y)

        self.pool.append(new_classifier)
        if len(self.pool) > self.n_classifiers:
            self.pool.pop(0)

        return self

    def predict(self, X):
        predictions = []
        for clf in self.pool:
            predictions.append(clf.predict(X))

        return stats.mode(predictions)[0]


stream = StreamGenerator(
    chunk_size=200,
    n_chunks=300,
    n_drifts=3,
    concept_sigmoid_spacing=999,
    weights=[0.05, 0.95],
)
clf_1 = GaussianNB()
clf_2 = EnsembleClassifier(GaussianNB(), 10)

clfs = (clf_1, clf_2)
evaluator = TestThenTrain(metrics=(balanced_accuracy_score))
evaluator.process(stream, clfs)

scores = evaluator.scores
clfs = ["GaussianNB", "EnsembleClf"]

plt.figure(figsize=(6, 3))

for score, clf in zip(scores, clfs):
    plt.plot(score, label=clf)

plt.title("Gaussian and MLP Comparison")
plt.ylim(0, 1)
plt.ylabel("Balanced accuracy score")
plt.xlabel("Chunk")
plt.grid(True)
plt.ylim(0.3, 1.0)

plt.legend()
plt.show()
