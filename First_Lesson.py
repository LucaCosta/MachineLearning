# Course Machine Learning by Alura
# features (1 sim, 0 não)
# long hair?
# small leg?
# bark?
pork1 = [0, 1, 0]
pork2 = [0, 1, 1]
pork3 = [1, 1, 0]

dog1 = [0, 1, 1]
dog2 = [1, 0, 1]
dog3 = [1, 1, 1]

# 1 => pork, 0 => dog
train_x = [pork1, pork2, pork3, dog1, dog2, dog3]
train_y = [1,1,1,0,0,0] # labels

from sklearn.svm import LinearSVC

model = LinearSVC()
model.fit(train_x, train_y)

mysterious_animal = [1,1,1]
model.predict([mysterious_animal])

mystery1 = [1,1,1]
mystery2 = [1,1,0]
mystery3 = [0,1,1]

test_x = [mystery1, mystery2, mystery3]
test_y = [0, 1, 1]

prediction = model.predict(teste_x)

correct = (prediction == test_y).sum()
total = len(test_x)
sucess_rate = correct/total
print("Sucess rate is: %.2f" % (sucess_rate * 100))

from sklearn.metrics import accuracy_score

sucess_rate = accuracy_score(test_y, prediction)
print("Taxa de acerto %.2f" % (sucess_rate * 100))
