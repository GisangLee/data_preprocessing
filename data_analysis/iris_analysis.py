from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import mglearn
import pandas as pd
import numpy as np

iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)


def summary_of_data(iris):
    print("Data Frame:\n {}\n".format(df))
    print("iris 데이터셋의 키 : {}\n".format(iris.keys()))
    print("Target Name: {}\n".format(iris['target_names']))
    print("Feature Name: {}\n".format(iris['feature_names']))
    print("Data Type: {}\n".format(type(iris['data'])))
    print("Data Size: {}\n".format(iris['data'].shape))
    print("Data의 처음 다섯 행 :\n {}\n".format(iris['data'][:5]))
    print("Target Type : {}\n".format(iris.target))
    print("Target Size : {}\n".format(iris.target.shape))


def get_model_dataframe(x_train, feature_names, y_train):
    iris_dataframe = pd.DataFrame(x_train, columns=feature_names)
    pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
    plt.show()


X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
print("X_train의 크기 : {}\n".format(X_train))
print("Y_train의 크기 : {}\n".format(y_train))
print("X_test의 크기 : {}\n".format(X_test))
print("Y_test의 크기 : {}\n".format(y_test))

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new Shape : {}\n".format(X_new.shape))

prediction = knn.predict(X_new)
print("예측 : {}\n".format(prediction))
print("예측한 타겟 이름 : {}\n".format(iris.target_names[prediction]))

y_pred = knn.predict(X_test)
print("실제 테스트 세트의 예측값: {}\n".format(y_pred))
print("실제 테스트 세트의 정확도: {:.2f}%\n".format(np.mean(y_pred == y_test)*100))
