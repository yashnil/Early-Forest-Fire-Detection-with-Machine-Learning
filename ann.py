# MLP Classifier

import sklearn
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(random_state=10)

mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)


accuracy = metrics.accuracy_score(y_test, y_pred) # finding the accuracy
score = f1_score(y_test, y_pred)

print(accuracy)
print(score)