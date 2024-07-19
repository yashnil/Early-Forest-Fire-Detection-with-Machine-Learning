
# Import Necessary Scikit-Learn Libraries

from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score

# logistic regression

model = linear_model.LogisticRegression()

print(y_train)

model.fit(X_train, y_train)

preds = model.predict(X_test)

accuracy = accuracy_score(y_test, preds)
score = f1_score(y_test, preds)

print(accuracy)
print(score)