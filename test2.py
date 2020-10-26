import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

# Assign colum names to the dataset
names = ['target', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash  ', 'Magnesium'
    , 'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity'
    , 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url, names=names)

print(dataset.head())
x1 = pd.DataFrame(data=dataset.iloc[:, 1:13].values,
                  columns=['Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash  ', 'Magnesium'
                      , 'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins'
                      , 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines'])
x2 = pd.DataFrame(data=dataset.iloc[:, 13].values,
                  columns=['Proline'])
X = pd.concat([x1, x2], axis=1).astype(float)
y = dataset.iloc[:, 0].values

print("x= \n", X.iloc[:,12])
print("y= \n", y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))