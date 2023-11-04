from sklearn.datasets import make_classification
from numpy import savetxt, column_stack
from matplotlib.pyplot import scatter, show

X, y = make_classification(400, n_informative=2, n_redundant=0, n_repeated=0, flip_y=0.08)

first_two_atrs = X[:, :2]
labels = y

columns = column_stack((first_two_atrs, labels))
savetxt('columns.csv', columns)

first_atr = X[:, 0]
second_atr = X[:, 1]

scatter(first_atr, second_atr, c=y)
show()