import numpy as np
import pandas as pd
import csv
import random
from datetime import date
import matplotlib.pyplot as plt

def create_table():
    with open('family_m_ru.txt', 'r', encoding='utf-8') as f:
        SurnamesMales = [i.rstrip() for i in f]

    with open('family_f_ru.txt', 'r', encoding='utf-8') as f:
        SurnamesFemales = [i.rstrip() for i in f]

    DepartmentsList = {
        "Отдел мобильной разработки": ["Тимлид", "Backend", "Frontend", "Тестировщик"],
        "Отдел веб-разработки": ["Тимлид", "Backend", "Frontend", "Тестировщик"],
        "Отдел desktop-приложений": ["Тимлид", "Backend", "Frontend", "Тестировщик"],
    }

    with open("Company_employees.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            (
            "Табельный номер", 
            "Имя сотрудника", 
            "Пол", 
            "Год рождения", 
            "Год трудоустойства", 
            "Подразделение",
            "Должность", 
            "Оклад", 
            "Выполненные проекты"
            )
        )
    
    def guess_letter():
        return (random.choice('АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯ') + ". " + random.choice('АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯ') + ".")

    #Генерация таблицы сотрудников
    for Employees in range(random.randint(1000,1500)):

        Department = random.choice(list(DepartmentsList))
        Gender = random.choice(["Мужчина","Женщина"] )
        BirthYear = random.randint(1972,2004)
        EmployenmtYear = min(date.today().year,BirthYear + 18)
        Salary =  random.randrange(40000,150000,1000)
        ProjectsAmount = random.randint(0,40)

        if (Gender == "Мужчина"):
            Name = SurnamesMales[random.randint(0,len(SurnamesMales)-1)]
        else: 
            Name = SurnamesFemales[random.randint(0,249)]
        

        with open("Company_employees.csv", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile )
            writer.writerow(
                [
                Employees + 1,
                (Name + " " + guess_letter()), 
                Gender, 
                BirthYear, 
                EmployenmtYear,
                Department, 
                random.choice(DepartmentsList[Department]), 
                Salary, 
                ProjectsAmount
                ]
                )

def create_numpy_stats():
    with open("Company_employees.csv") as csvfile:
        metadata = [list(row) for row in csv.reader(csvfile)]

    data = np.array(metadata)
    genders = data[:, 2]  
    departments = data[:, 5]
    salary = data[1:, 7].astype("int32")
    emp_year = data[1:, 4].astype("int32")

    countsmode_salary = np.bincount(salary)
    countsmode_emp_year = np.bincount(emp_year)

    print("###Статистика Numpy###")
    print('')
    print("###Статистика по сотрудникам###")
    print("Количество сотрудников всего:", np.count_nonzero(salary))
    print("Доля сотрудников женского пола:", round(np.sum(genders == "Женщина") / np.size(genders),3))
    print("Больше всего новых сотрудников в", np.argmax(countsmode_emp_year), "году")
    print('')
    print("###Статистика по зарплате###")
    print("Максимальная з/п:", np.max(salary))
    print("Минимальная з/п:", np.min(salary))
    print("Средняя з/п:", round(np.average(salary),3))
    print("Дисперсия з/п:", round(np.var(salary),3))
    print("Ст. откл. кол-ва зарплаты:", round(np.std(salary),3))
    print("Медиана кол-ва зарплаты:", np.median(salary))
    print("Мода зарплаты:", np.argmax(countsmode_salary))
    print('')
    print("###Статистика по подразделениям###")
    print("Количество сотрудников в отделе мобильной разработки:", np.count_nonzero(departments == "Отдел мобильной разработки"))
    print("Количество сотрудников в отделе веб-разработки:", np.count_nonzero(departments == "Отдел веб-разработки"))
    print("Количество сотрудников в отделе desktop-приложений:", np.count_nonzero(departments == "Отдел desktop-приложений"))
    print("Больше всего сотрудников в отделе:", "Отдел мобильной разработки" if ((np.count_nonzero(departments == "Отдел мобильной разработки")\
        >np.count_nonzero(departments == "Отдел веб-разработки"))and(np.count_nonzero(departments == "Отдел мобильной разработки")\
        >np.count_nonzero(departments == "Отдел desktop-приложений"))) else ( "Отдел веб-разработки" if (np.count_nonzero(departments == "Отдел веб-разработки")\
            >np.count_nonzero(departments == "Отдел desktop-приложений")) else "Отдел desktop-приложений"))
    print('')

def create_pandas_stats():
    data = pd.read_csv("Company_employees.csv", encoding='cp1251')
  #  genders = data[2]
  #  departments = data[5]
   # salary = data[7]
   # emp_year = data[4]

    print("###Статистика Pandas###")
    print('')
    print("###Статистика по сотрудникам###")
    print("Количество сотрудников всего:", data["Табельный номер"].count())
    print("Доля сотрудников женского пола:", round(data["Пол"].value_counts()["Женщина"] / data["Пол"].shape[0],3))
    print("Больше всего новых сотрудников в", data["Год трудоустойства"].mode()[0], "году")
    print('')
    print("###Статистика по зарплате###")
    print("Максимальная з/п:", data["Оклад"].max())
    print("Минимальная з/п:", data["Оклад"].min())
    print("Средняя з/п:", round(data["Оклад"].mean(),3))
    print("Дисперсия з/п:", round(data["Оклад"].var(),3))
    print("Ст. откл. кол-ва зарплаты:", round(data["Оклад"].std(),3))
    print("Медиана кол-ва зарплаты:", data["Оклад"].median())
    print("Мода зарплаты:", data["Оклад"].mode()[0])
    print('')
    print("###Статистика по подразделениям###")
    print("Количество сотрудников в отделе мобильной разработки:", (data["Подразделение"].value_counts()["Отдел мобильной разработки"]))
    print("Количество сотрудников в отделе веб-разработки:", (data["Подразделение"].value_counts()["Отдел веб-разработки"]))
    print("Количество сотрудников в отделе desktop-приложений:", (data["Подразделение"].value_counts()["Отдел desktop-приложений"]))
    print("Больше всего сотрудников в отделе:", "Отдел мобильной разработки" if ((data["Подразделение"].value_counts()["Отдел мобильной разработки"]\
        >data["Подразделение"].value_counts()["Отдел веб-разработки"])and(data["Подразделение"].value_counts()["Отдел мобильной разработки"]\
        >data["Подразделение"].value_counts()["Отдел desktop-приложений"])) else ( "Отдел веб-разработки" if (data["Подразделение"].value_counts()["Отдел веб-разработки"]\
            >data["Подразделение"].value_counts()["Отдел desktop-приложений"]) else "Отдел desktop-приложений"))

def create_graphics():
    data = pd.read_csv("Company_employees.csv", encoding='cp1251')
    proportionPosProj = {}
    positions = data['Должность'].unique()

    for item in positions:
        pos = data[data['Должность'] == item]
        proportionPosProj[item] = round(pos['Выполненные проекты'].sum() / pos['Выполненные проекты'].count(), 2)

    genders=[data["Пол"].value_counts()["Мужчина"],data["Пол"].value_counts()["Женщина"]]
    plt.pie(genders, labels=["Мужчины", "Женщины"])
    plt.title('Соотношение мужчин и женщин в компании', loc='center')
    plt.show()

    plt.hist(data['Подразделение'], bins=15)
    plt.title('Количество сотрудников по отделам', loc='center')
    plt.show()

    plt.bar(proportionPosProj.keys(), proportionPosProj.values())
    plt.title('Соотношение количества проектов по должностям', loc='center')
    plt.show()

    
create_table()
create_numpy_stats()
create_pandas_stats()
create_graphics()
