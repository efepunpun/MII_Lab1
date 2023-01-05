import matplotlib.pyplot as plt
import pandas as pd
import pylab

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris


def DTC(X_train_d, X_test_d, y_train_d, y_test_d):
    Tree_model = DecisionTreeClassifier(max_depth=12)
    Tree_model.fit(X_train_d, y_train_d)
    y_pred_d = Tree_model.predict(X_test_d)

    print('Характеристики классификации DT:')
    print(classification_report(y_test_d, y_pred_d, zero_division=0))
    print('Точность классификатора деревом решений: ', accuracy_score(y_test_d, y_pred_d))
    print("Матрица ошибок:\n ", confusion_matrix(y_test_d, y_pred_d))

    pylab.subplot(1, 2, 1)
    plt.pie(pd.Series(y_test_d).value_counts().sort_index(), labels = sorted(pd.Series(y_test_d).unique()), autopct='%1.1f%%')
    plt.title('Оригинальные данные')
    pylab.subplot(1, 2, 2)
    plt.pie(pd.Series(y_pred_d).value_counts().sort_index(), labels = sorted(pd.Series(y_pred_d).unique()), autopct='%1.1f%%')
    plt.title('Предсказание DT')
    plt.show()


def SGDC(X_train_l, X_test_l, y_train_l, y_test_l):
    Linear_model = SGDClassifier()
    Linear_model.fit(X_train_l, y_train_l)
    y_pred_l = Linear_model.predict(X_test_l)

    print('Характеристики линейной классификации с SGD-обучением:')
    print(classification_report(y_test_l, y_pred_l, zero_division=0))
    print('Точность линейного классификатора с SGD-обучением: ', accuracy_score(y_test_l, y_pred_l))
    print("Матрица ошибок:\n ", confusion_matrix(y_test_l, y_pred_l))

    pylab.subplot(1, 2, 1)
    plt.pie(pd.Series(y_test_l).value_counts().sort_index(), labels = sorted(pd.Series(y_test_l).unique()), autopct='%1.1f%%')
    plt.title('Оригинальные данные')
    pylab.subplot(1, 2, 2)
    plt.pie(pd.Series(y_pred_l).value_counts().sort_index(), labels = sorted(pd.Series(y_pred_l).unique()), autopct='%1.1f%%')
    plt.title('Предсказание SGD')
    plt.show()

def GPC(X_train_g, X_test_g, y_train_g, y_test_g):
    Gaussian_model = GaussianProcessClassifier()
    Gaussian_model.fit(X_train_g, y_train_g)
    y_pred_g = Gaussian_model.predict(X_test_g)

    print('Характеристики классификации гауссовским процессом:')
    print(classification_report(y_test_g, y_pred_g, zero_division=0))
    print('Точность классификатора гауссовским процессом: ', accuracy_score(y_test_g, y_pred_g))
    print("Матрица ошибок:\n ", confusion_matrix(y_test_g, y_pred_g))

    pylab.subplot(1, 2, 1)
    plt.pie(pd.Series(y_test_g).value_counts().sort_index(), labels = sorted(pd.Series(y_test_g).unique()), autopct='%1.1f%%')
    plt.title('Оригинальные данные')
    pylab.subplot(1, 2, 2)
    plt.pie(pd.Series(y_pred_g).value_counts().sort_index(), labels = sorted(pd.Series(y_pred_g).unique()), autopct='%1.1f%%')
    plt.title('Предсказание GP')
    plt.show()

iris_dataset = load_iris()
X, y = iris_dataset.data, iris_dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

DTC(X_train, X_test, y_train, y_test)
SGDC(X_train, X_test, y_train, y_test)
GPC(X_train, X_test, y_train, y_test)