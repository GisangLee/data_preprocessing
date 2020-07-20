import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
import graphviz

os.environ['PATH'] += os.pathsep + 'C:\Program Files (x86)/Graphviz2.38/bin/'

cancer = load_breast_cancer()
print(cancer.keys())

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

dTreeAll = DecisionTreeClassifier(max_depth=10, random_state=0)

dTreeAll.fit(X_train, y_train)

export_graphviz(dTreeAll, out_file="dicisiontree.dot", class_names=['malignant', 'benign'],
                feature_names=cancer.feature_names, impurity=False, filled=True)

print("Score : {:.2f}%\n".format(dTreeAll.score(X_train, y_train)*100))

y_predict = dTreeAll.predict(X_test)
print("테스트 정확도 : {:.2f}%\n".format(accuracy_score(y_test, y_predict)*100))

with open("dicisiontree.dot") as f:
    dot_graph = f.read()
dot = graphviz.Source(dot_graph)
dot.format = "png"
dot.render(filename="dicisiontree", directory="images/decision_trees", cleanup=True)
dot
