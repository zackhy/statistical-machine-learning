import numpy as np
from models.naive_bayes import BernoulliNB as MyBNB
from models.naive_bayes import MultinomialNB as MyMNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

X = np.random.randint(5, size=(6, 100))
y = np.array([1, 2, 3, 4, 4, 5])

my_clf = MyBNB().fit(X, y)
my_score = my_clf.score(X, y)

clf = BernoulliNB().fit(X, y)
score = clf.score(X, y)

print('My Bernoulli score: {:.3f}\nSklearn Bernoulli score: {:.3f}'.format(my_score, score))

my_clf = MyMNB().fit(X, y)
my_score = my_clf.score(X, y)

clf = MultinomialNB().fit(X, y)
score = clf.score(X, y)

print('My Multinomial score: {:.3f}\nSklearn Multinomial score: {:.3f}'.format(my_score, score))
