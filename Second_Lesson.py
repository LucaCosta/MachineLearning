# Course Machine Learning by Alura

import pandas as pd

uri = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"
data = pd.read_csv(uri)
data.head()

x = data[["home","how_it_works","contact"]]
x.head()

y = data["bought"]
y.head()

data.shape

train_x = x[:75]
train_y = y[:75]
test_x = x[75:]
test_y = y[75:]

print("We will train with %d elements and test with %d elements" % (len(train_x), len(test_x)))

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

model = LinearSVC()
model.fit(train_x, train_y)
predction = model.predict(test_x)

accuracy = accuracy_score(test_y, prediction) * 100
print("The accuracy was %.2f%%" % accuracy)

"""# Using the library to train and test separate"""

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

SEED = 20

train_x, test_x, train_y, test_y = train_test_split(x, y, random_state = SEED, test_size = 0.25)
print("We will train with %d elements and test with %d elements" % (len(train_x), len(test_x)))

model = LinearSVC()
model.fit(train_x, train_y)
prediction = model.predict(test_x)

accuracy = accuracy_score(test_y, prediction) * 100
print("The accuracy was %.2f%%" % accuracy)

train_y.value_counts()

test_y.value_counts()

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

SEED = 20

train_x, test_x, train_y, test_y = train_test_split(x, y,
                                                         random_state = SEED, test_size = 0.25,
                                                         stratify = y)
print("We will train witg %d elements and test with %d elements" % (len(train_x), len(test_x)))

model = LinearSVC()
model.fit(train_x, train_y)
prediction = model.predict(test_x)

accuracy = accuracy_score(test_y, prediction) * 100
print("The accuracy was %.2f%%" % accuracy)

train_y.value_counts()

test_y.value_counts()
