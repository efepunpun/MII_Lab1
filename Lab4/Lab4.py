import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from collections import Counter

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

def distance_square(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

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

def knn_scikit(k, X, y):

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)


    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(x_train, y_train)


    y_pred = classifier.predict(x_test)

    return x_train, x_test, y_test, y_test, y_pred

def gen_classification(dataset):
    X = dataset.iloc[:, 1:3]
    y = dataset.iloc[:, 3]

    x_train = dataset.iloc[:(int)(0.5 * len(dataset)), 1:3]
    x_test = dataset.iloc[(int)(0.5 * len(dataset)):, 1:3]
    y_test = dataset.iloc[(int)(0.5 * len(dataset)):, 3]

    y_pred = knn(7, x_train, x_test, y)
    X_train, X_test, Y_test, Y_test, Y_pred = knn_scikit(7, X, y)
    
    print('Статистика KNN')
    print(classification_report(y_test, y_pred))

    print('Статистика KNN sklearn')
    print(classification_report(Y_test, Y_pred))

   
    return x_test, X_test, y_pred, Y_pred

def visualize_stats(x_test, X_test, color, color_scikit):
    f, ax = plt.subplots(2, 1, figsize=(8, 8))

    ax[0].scatter(x_test['Sweet'][:], x_test['Crunch'][:], c=color)
    ax[0].set_title('Статистика KNN')
    

    ax[1].scatter(X_test[:, 0], X_test[:, 1], c=color_scikit)
    ax[1].set_title('Статистика KNN sklearn')
   
    plt.show()

def create_stats():
    create_table('false')
    data = pd.read_csv('Products.csv')
    x_test, X_test, y_pred, Y_pred = gen_classification(data)

    color = [label.replace("Fruit", "orange", 1)
                    .replace("Veggy", "green", 1)
                    .replace("Protein", "brown", 1) for label in y_pred]

    color_scikit = [label.replace("Fruit", "orange", 1)
                        .replace("Veggy", "green", 1)
                        .replace("Protein", "brown", 1) for label in Y_pred]


    visualize_stats(x_test, X_test, color, color_scikit)

def create_more_stats():
    create_table('true')
    more_data = pd.read_csv('Products.csv')
    x_test, X_test, y_pred, Y_pred = gen_classification(more_data)

    color = [label.replace("Fruit", "orange", 1)
                        .replace("Veggy", "green", 1)
                        .replace("Protein", "brown", 1)
                        .replace("Berry", "red", 1) for label in y_pred]

    color_scikit = [label.replace("Fruit", "orange", 1)
                                .replace("Veggy", "green", 1)
                                .replace("Protein", "brown", 1)
                                .replace("Berry", "red", 1) for label in Y_pred]

    visualize_stats(x_test, X_test, color, color_scikit)

create_stats()
create_more_stats()