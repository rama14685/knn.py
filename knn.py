# Import library
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets


wine = datasets.load_wine()
X = wine.data
y = wine.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


knn = KNeighborsClassifier(n_neighbors=3)


knn.fit(X_train, y_train)


accuracy = knn.score(X_test, y_test)
print(f'Akurasi: {accuracy}')

import joblib

joblib.dump(knn, 'knn_model_wine.joblib')
