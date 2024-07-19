import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import datasets  # to retrieve the iris Dataset
from sklearn.preprocessing import StandardScaler  # to standardize the features
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA  # to apply PCA
from sklearn import metrics
from google.colab import drive

drive.mount('/content/drive')

originalAlgerianDataSet = pd.read_csv("/content/drive/MyDrive/Datasets/Algerian_forest_fires_dataset4.csv")
algerianDataSet = originalAlgerianDataSet.copy()

for i in range(len(originalAlgerianDataSet)):
  if originalAlgerianDataSet['Classes  '][i] == 'not fire   ':
    algerianDataSet['Classes  '][i] = 0
  else:
    algerianDataSet['Classes  '][i] = 1

algerianDataSet = algerianDataSet.drop('day', axis=1)

algerianDataSet.rename(columns={' Ws':'Wind'}, inplace=True)
algerianDataSet.rename(columns={' RH':'RH'}, inplace=True)
algerianDataSet.rename(columns={'Rain ':'Rain'}, inplace=True)
algerianDataSet.rename(columns={'Classes  ':'Class'}, inplace=True)
algerianDataSet.rename(columns={'day':'Day'}, inplace=True)
algerianDataSet.rename(columns={'month':'Month'}, inplace=True)
algerianDataSet.rename(columns={'year':'Year'}, inplace=True)

newDf = algerianDataSet.drop('FFMC', axis=1)
newDf = newDf.drop('DMC', axis=1)
newDf = newDf.drop('DC', axis=1)
newDf = newDf.drop('ISI', axis=1)
newDf = newDf.drop('BUI', axis=1)
newDf = newDf.drop('FWI', axis=1)
newDf = newDf.drop('Year', axis=1)

# visualize it with a histogram
plt.hist(newDf['Class'])
plt.show()
newDf.hist()

X_data = newDf.drop('Class', axis=1)
Y_data = newDf[['Class']].copy().astype('int')

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3, random_state=42)

scaler = StandardScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(X_data)) #scaling the data
scaled_data


clf = RandomForestClassifier(random_state=42,min_samples_split=20,min_samples_leaf=6)
# optimal performance accuracy is min_samples_leaf = 20

# Hyperparameter Tuning: n_estimators=50,max_depth=15,min_samples_split=9,max_features=1,min_samples_leaf=1

# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)

# performing predictions on the test dataset
y_pred = clf.predict(X_test)

# metrics are used to find accuracy or error
print()

# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
print("F1 SCORE OF THE MODEL: ", metrics.f1_score(y_test, y_pred))
print("PRECISION SCORE OF THE MODEL: ", metrics.precision_score(y_test, y_pred))
print("RECALL SCORE OF THE MODEL: ", metrics.recall_score(y_test, y_pred))

print(y_test)
print(y_pred)

y_pred_proba = clf.predict_proba(X_test)[::,1]

print(y_pred_proba)