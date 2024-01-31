import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from strlearn.evaluators import TestThenTrain
from strlearn.metrics import balanced_accuracy_score
from strlearn.streams import StreamGenerator

stream = StreamGenerator(
    chunk_size=200,
    n_chunks=300,
    n_drifts=3,
    concept_sigmoid_spacing=999,
    weights=[0.05, 0.95],
)

clf_1 = GaussianNB()
clf_2 = MLPClassifier()
clfs = (clf_1, clf_2)

evaluator = TestThenTrain(metrics=(balanced_accuracy_score))
evaluator.process(stream, clfs)

scores = evaluator.scores
clfs = ["GaussianNB", "MLPClassifier"]

plt.figure(figsize=(6, 3))

for score, clf in zip(scores, clfs):
    plt.plot(score, label=clf)

plt.title("Gaussian and MLP Comparison")
plt.ylim(0, 1)
plt.ylabel("Balanced accuracy score")
plt.xlabel("Chunk")
plt.grid(True)
plt.ylim(0.4, 1.0)

plt.legend()
plt.show()
