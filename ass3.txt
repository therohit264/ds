import pandas as pd
df = pd.read_csv("Employee.csv")
print('Employee_Salary Dataset is successfully loaded!!')
df2 = pd.read_csv("Iris.csv")
print('Iris Dataset is successfully loaded!!')

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

choice = 1
while(choice != 9):
    print('--------------MENU-----------')
    print('1. Display information of employee salary dataset')
    print('2. Display statistical information of numerical columns of employee salary dataset')
    print('3. Groupwise statistical of employee salary dataset')
    print('4. Bar plot for all statistics of employee salary dataset')
    print('5. Display information of Iris dataset')
    print('6. Display statistical information of numerical columns of iris dataset')
    print('7. Groupwise statistical of iris dataset')
    print('8. Bar plot for all statistics of iris dataset')
    print('9. Exit')
    choice = int(input("Enter your choice: "))
    if choice == 1:
        print('Information of dataset:\n', df.info)
        print('Shape of Dataset (row x column): ', df.shape)
        print('Columns Name: ', df.columns)
        print('Total elements in dataset: ', df.size)
        print('Datatypes of attributes (columns): ', df.dtypes)
        print('First 5 rows:\n', df.head(5))
        print('Last 5 rows:\n', df.tail(5))
        print('Any 5 rows:\n', df.sample(5))

    if choice == 2:
        print('Statistical information of numerical columns: \n')  
        print(df.describe())

    if choice == 3:
        print('Groupwise statistical summary..')
        columns = ['ExperienceInCurrentDomain', 'Age', 'JoiningYear']
        for column in columns:
                print('\n---------------', column, '--------------\n')
                print(df.groupby('Gender')[column].describe())

    if choice == 4:
            features = ['JoiningYear', 'Age', 'ExperienceInCurrentDomain']
            for var in features:
                    df_stat = df.groupby('Gender')[var].describe()
                    df_stat = df_stat[['min', 'max', 'mean', '50%', 'std']]
                    df_stat.columns = ['Min', 'Max', 'Mean', 'Median', 'STD']
                    
                    df_stat.plot(kind='bar', figsize=(10, 6))
                    plt.xlabel('Statistical Information')
                    plt.ylabel(var)
                    plt.title(f'Groupwise statistical information of {var} by Gender')
                    plt.legend(title='Gender')
                    plt.show()

    if choice == 5:
        df2 = pd.read_csv('Iris.csv')

        print('Information of dataset:\n', df2.info)
        print('Shape of Dataset (row x column): ', df2.shape)
        print('Columns Name: ', df2.columns)
        print('Total elements in dataset: ', df2.size)
        print('Datatypes of attributes (columns): ', df2.dtypes)
        print('First 5 rows:\n', df2.head().T)
        print('Last 5 rows:\n', df2.tail().T)
        print('Any 5 rows:\n', df2.sample(5).T)

    if choice == 6:
        print('Statistical information of numerical columns: \n')
        columns = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
        for column in columns:
             print(df2.describe())

    if choice == 7:
        print('Groupwise statistical summary..')
        columns = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
        for column in columns:
                       print(df2.groupby('Species')[column].describe())


    if choice == 8:
       X = ['Min', 'Max', 'Mean', 'Median', 'STD']
       features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

       for var in features:
            fig, ax = plt.subplots()
            grouped_stats = df2.groupby('Species')[var].agg(['min', 'max', 'mean', 'median', 'std'])
            grouped_stats.plot(kind='bar', ax=ax, width=0.25)
            ax.set_xlabel('Statistical Information')
            ax.set_ylabel(var)
            ax.set_title(f'Groupwise statistical information of {var} by Species')
            plt.legend(title='Species')
            plt.show()

    if choice == 9:
        break