import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.decomposition import PCA
import warnings
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

warnings.simplefilter(action='ignore', category=FutureWarning)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

# loading dataset into Pandas DataFrame
df = pd.read_csv(url, names=['target', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash  ', 'Magnesium'
    , 'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity'
    , 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'])
print(df.head())

features = ['Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash  ', 'Magnesium'
    , 'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins'
    , 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline']
x = df.loc[:, features].values
y = df.loc[:, ['target']].values
x = StandardScaler().fit_transform(x)
print(pd.DataFrame(data=x, columns=features).head())
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents,
                           columns=['principal component 1', 'principal component 2'])
print(principalDf.head(5))
print(df[['target']].head())
finalDf = pd.concat([principalDf, df[['target']]], axis=1)
print(finalDf.head(5))

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_zlabel('Principal Component 3', fontsize=15)
ax.set_title('3 Component PCA', fontsize=20)

targets = [1, 2, 3]
colors = ['r', 'g', 'b']

for target, color in zip(targets, colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c=color
               , s=50)
ax.legend(targets)
ax.grid()
plt.show()
print(pca.explained_variance_ratio_)



X = finalDf.iloc[:, :-1].values
y = finalDf.iloc[:, 2].values
"""""
X = finalDf.iloc[:, 0:1].values
y = finalDf.iloc[:, 1].values
"""""

print("x= \n", X)
print("y= \n", y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

error = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

t = PrettyTable(['K', 'Error'])
for i in range(1, 39):
    t.add_row([i, error[i]])

print(t)


