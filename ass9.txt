import pandas as pd
df = pd.read_csv("titanic_data.csv")
print('Titanic Dataset is successfully loaded....')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
choice = 1
while(choice != 10):
    print('--------Menu----------')
    print('1.Display information of dataset')
    print('2.Find Missing values')
    print('3.Fill Missing values')
    print('4.Box Plot of 1-variable (Age & Fare)') 
    print('5.Box Plot of 2-variables')
    print('6.Box Plot of 3-variables')
    print('7. Exit')
    choice = int(input('Enter your choice: '))
    if choice==1:
        print('Information of Dataset:\n', df.info)
        print('Shape of Dataset (row x column):', df.shape)
        print('Columns Name:', df.columns)
        print('Total elements in dataset:', df.size)
        print('Datatype of attributes (columns):', df.dtypes)
        print('First 5 rows:\n', df.head().T)
        print('Last 5 rows:\n', df.tail().T) 
        print('Any 5 rows:\n',df.sample(5).T)

    if choice == 2:
        print('Total Number of Null Values in Dataset:', df.isna().sum())

    if choice == 3:
        df['Age'].fillna(df['Age'].median(), inplace = True)
        print('Null values are: \n',df.isna().sum())

    if choice == 4:
        fig, axes = plt.subplots(1,2)
        fig.suptitle('Boxplot of 1-variables (Age & Fare)')
        sns.boxplot(data = df, x = 'Age', ax=axes[0]) 
        sns.boxplot(data = df, x = 'Fare', ax=axes[1])
        plt.show()

    if choice == 5:
        fig, axes = plt.subplots(2,2) 
        fig.suptitle('Boxplot of 2-variables')
        sns.boxplot(data = df, x = 'Embarked', y='Age', ax=axes [0,0]) 
        sns.boxplot(data = df, x = 'Embarked', y='Fare', ax=axes [0,1])
        sns.boxplot(data = df, x = 'Age', y='Age', ax=axes [1,0]) 
        sns.boxplot(data = df, x = 'Sex', y='Fare', ax=axes [1,1])
        fig.tight_layout()
        plt.show()

    if choice == 6:
        fig, axes = plt.subplots(1,2)
        fig.suptitle('Boxplot of 3-variables')
        sns.boxplot(data = df, x ='Sex', y='Age', hue ='Embarked', ax=axes[0]) 
        sns.boxplot(data = df, x='Sex', y='Fare', hue = 'Embarked', ax=axes[1])
        fig.tight_layout()
        plt.show()

    if(choice == 7):
	    break