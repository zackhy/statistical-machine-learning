import numpy as np
from models.naive_bayes import BernoulliNB as MyNB
from sklearn.naive_bayes import BernoulliNB

X = np.random.randint(2, size=(6, 100))
y = np.array([1, 2, 3, 4, 4, 5])

my_clf = MyNB().fit(X, y)
my_score = my_clf.score(X, y)

clf = BernoulliNB().fit(X, y)
score = clf.score(X, y)

print('My score: {:.3f}\nSklearn score: {:.3f}'.format(my_score, score))
