import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import random
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from collections import Counter

#Генерация данных
def create_table(isExtended):
    Classes = ["Fruit", "Protein", "Veggy"]
    ExtendedClasses = ["Fruit", "Protein", "Veggy","Berry"]
    with open('fruits.txt', 'r', encoding='utf-8') as f:
            FruitList = [i.rstrip() for i in f]
    with open('veggys.txt', 'r', encoding='utf-8') as f:
            VeggyList = [i.rstrip() for i in f]
    with open('proteins.txt', 'r', encoding='utf-8') as f:
            ProteinList = [i.rstrip() for i in f]  
    with open('berries.txt', 'r', encoding='utf-8') as f:
            BerryList = [i.rstrip() for i in f]     
    with open("Products.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                (
                "Product", 
                "Sweet", 
                "Crunch", 
                "Class"  
                )
            )
    for ProductList in range (100):
        if (isExtended == 'true'):
            Class = random.choice(ExtendedClasses)
        else:
            Class = random.choice(Classes)
        Sweet = -1
        Crunch = -1
        Product = ""
        if (Class == "Fruit"):
                Sweet = random.randint(5,10)
                Crunch = random.randint(3,8)
                Product = random.choice(FruitList)
        elif (Class == "Protein"):
                Sweet = random.randint(1,2)
                Crunch = random.randint(1,5)
                Product = random.choice(ProteinList)
        elif (Class == "Veggy"):
                Sweet = random.randint(2,5)
                Crunch = random.randint(5,10)
                Product = random.choice(VeggyList)
        elif (Class == "Berry" and isExtended == 'true'):
                Sweet = random.randint(6,10)
                Crunch = random.randint(1,5)
                Product = random.choice(BerryList)
        with open("Products.csv", 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [    
                        Product,
                        Sweet,
                        Crunch,
                        Class
                    ]
                    )
        
#Евклидово расстояние
def distance_square(data1, data2):
    distance = 0
    for i in range(len(data1)-1):
        distance += (data1[i] - data2[i])**2
    return math.sqrt(distance)

#Метод KNN без библиотеки sklearn
def knn(k, x_train, x_test, y):
    y_pred = []
    for i in range(len(x_test)):
        distances = []
        for j in range(len(x_train)):
            dist = distance_square(np.array(x_train)[j, :], np.array(x_test)[i])
            distances.append(dist)
        distances = np.array(distances)
        k_distances = np.argsort(distances)[:k]
        values = y[k_distances]
        y_pred.append(Counter(values).most_common(1)[0][0])
    return y_pred

#Метод KNN с библиотекой sklearn
def knn_scikit(k, X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    ss = StandardScaler()
    ss.fit(x_train)
    x_train = ss.transform(x_train)
    x_test = ss.transform(x_test)
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    return x_train, x_test, y_test, y_test, y_pred

#Классификация полученных данных
def classify_data(dataset):
    y = dataset.iloc[:, 3]

    x_train = dataset.iloc[:(int)(0.5 * len(dataset)), 1:3]
    x_test = dataset.iloc[(int)(0.5 * len(dataset)):, 1:3]
    y_test = dataset.iloc[(int)(0.5 * len(dataset)):, 3]
    y_pred = knn(7, x_train, x_test, y)
    print('Статистика KNN')
    print(classification_report(y_test, y_pred))

    X = dataset.iloc[:, 1:3]
    X_train, X_test, Y_test, Y_test, Y_pred = knn_scikit(7, X, y)
    print('Статистика KNN sklearn')
    print(classification_report(Y_test, Y_pred))

    return x_test, X_test, y_pred, Y_pred

#Визуализация статистики
def visualize_stats(x_test, X_test, color, color_2):
    f, ax = plt.subplots(2, 1, figsize=(8, 8))
    ax[0].scatter(x_test['Sweet'][:], x_test['Crunch'][:], c = color)
    ax[0].set_title('Статистика KNN')
    ax[1].scatter(X_test[:, 0], X_test[:, 1], c = color_2)
    ax[1].set_title('Статистика KNN sklearn')
    plt.show()

#Генерация статистики на основе базовых данных
create_table('false')
data = pd.read_csv('Products.csv')
x_test, X_test, y_pred, Y_pred = classify_data(data)
color = [label.replace("Fruit", "yellow", 1).replace("Veggy", "green", 1).replace("Protein", "grey", 1) for label in y_pred]
color_2 = [label.replace("Fruit", "yellow", 1).replace("Veggy", "green", 1).replace("Protein", "grey", 1) for label in Y_pred]
visualize_stats(x_test, X_test, color, color_2)

#Генерация статистики на основе расширенных данных
create_table('true')
more_data = pd.read_csv('Products.csv')
x_test, X_test, y_pred, Y_pred = classify_data(more_data)
color = [label.replace("Berry", "purple", 1) for label in y_pred]
color_2 = [label.replace("Fruit", "yellow", 1).replace("Veggy", "green", 1).replace("Protein", "grey", 1).replace("Berry", "purple", 1) for label in Y_pred]
visualize_stats(x_test, X_test, color, color_2)

