import mglearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.datasets import load_breast_cancer
import numpy as np


def binary_classification():
    X, y = mglearn.datasets.make_forge()
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)

    font_path = 'C:/Windows/Fonts/malgun.ttf'
    font_name = fm.FontProperties(fname=font_path).get_name()
    mpl.rc('font', family=font_name)

    print("X.shape : {}\n".format(X.shape))
    mglearn.plots.plot_knn_classification(n_neighbors=3)

    plt.show()


def regression_classification():
    X, y = mglearn.datasets.make_wave(n_samples=40)
    plt.plot(X, y, 'o')
    plt.ylim(-3, 3)
    plt.xlabel("특성")
    plt.ylabel("타깃")
    plt.show()


def cancerVisualize():
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target,
                                                        random_state=66)
    training_accuracy = []
    test_accuracy = []
    neighbors_settings = range(1, 11)
    for n_neighbor in neighbors_settings:
        clf = KNeighborsClassifier(n_neighbors=n_neighbor)
        clf.fit(X_train, y_train)
        training_accuracy.append(clf.score(X_train, y_train))
        test_accuracy.append(clf.score(X_test, y_test))
    plt.plot(neighbors_settings, training_accuracy, label="Training Accuracy")
    plt.plot(neighbors_settings, test_accuracy, label="Test Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("n_neighbors")
    plt.legend()
    plt.show()
    print("cancer.keys() : {}\n".format(cancer.keys()))
    print("유방암 데이터 형태 : {}\n".format(cancer.data.shape))
    print("클래스별 샘플 개수 : {}\n".format({n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
    print("특성 이름 : {}\n".format(cancer.feature_names))


def KNN():
    X, y = mglearn.datasets.make_forge()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    print("테스트 세트 예측 : {}\n".format(clf.predict(X_test)))
    print("테스트 세트 정확도 : {:.2f}%\n".format(clf.score(X_test, y_test) * 100))

    fg, axes = plt.subplots(1, 3, figsize=(10, 3))
    for n_neighbors, ax in zip([1, 3, 9], axes):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
        mglearn.plots.plot_2d_separator(clf, X, fill=None, eps=0, ax=ax, alpha=.4)
        mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
        ax.set_title("{} Neighbors".format(n_neighbors))
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
    axes[0].legend(loc=3)
    plt.show()


cancerVisualize()
