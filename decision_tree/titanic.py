import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

os.environ['PATH'] += os.pathsep + 'C:\Program Files (x86)/Graphviz2.38/bin/'

data = pd.read_excel("../data_sets/titanic.xls")
print(data.info())
print(data.describe())

tmp = []
for each in data.sex:
    if each == 'female':
        tmp.append(1)
    elif each == 'male':
        tmp.append(0)

data.sex = tmp
data.survived = data.survived.astype('int')
data.pclass = data.pclass.astype('int')
data.sibsp = data.sibsp.astype('int')
data.parch = data.parch.astype('int')
data.fare = data.fare.astype('float')
data.sex = data.sex.astype('int')

data = data[data.age.notnull()]

print(data.info())
print(data)

to_use_data = data[['pclass', 'sex', 'age', 'sibsp', 'parch']]
print(to_use_data.head())

X_train, X_test, y_train, y_test = train_test_split(to_use_data, data[['survived']], random_state=13, test_size=0.2)

X_train = X_train.reset_index()
X_train = X_train.drop(['index'], axis=1)

X_test = X_test.reset_index()
X_test = X_test.drop(['index'], axis=1)

y_train = y_train.reset_index()
y_train = y_train.drop(['index'], axis=1)

y_test = y_test.reset_index()
y_test = y_test.drop(['index'], axis=1)

kn_clf = KNeighborsClassifier(n_neighbors=3)
kn_clf.fit(X_train, y_train)

dc_clf = DecisionTreeClassifier(max_depth=10)
dc_clf.fit(X_train, y_train)

print("KNeighbor 알고리즘 테스트 예측 : {}\n".format(kn_clf.predict(X_test)))
print("KNeighbor 알고리즘 테스트 정확도 : {:.2f}%\n".format(kn_clf.score(X_test, y_test)*100))

print("Decision Tree 알고리즘 테스트 예측: {}\n".format(dc_clf.predict(X_test)))
print("Decision Tree 알고리즘 테스트 정확도: {:.2f}%\n".format(dc_clf.score(X_test, y_test)*100))


