def DetectOutlier(df, var):
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    high, low = Q3+1.5*IQR, Q1-1.5*IQR

    print("Highest allowed in variable:", var, high) 
    print("lowest allowed in variable:", var, low) 
    count = df[(df[var] > high) | (df[var] < low)][var].count() 
    print('Total outliers in:', var,':', count)

    df = df[((df[var] >= low) & (df[var] <= high))]
    print('Outliers removed in', var)
    return df

import pandas as pd
df = pd.read_csv("student_dataset.csv")
print('Student Academic Performance Dataset is successfully loaded ....')

import seaborn as sns
import matplotlib.pyplot as plt

choice = 1
while(choice != 10):

    print('----------------MENU-----------------')
    print('1.Display information of dataset')
    print('2.Display Statistical information of Numerical Columns')
    print('3.Find and Fill the Missing values')
    print('4.Detect outliers')
    print('5.Data Transformation: Conversion of categorical to quantitative')
    print('6.Boxplot with 2 variables (gender and raisedhands)')
    print('7.Boxplot with 3 variables (gender, nationality, discussion)')
    print('8.Scatterplot to see relation between (raisedhands, VisitedResources)')
    print('9. Exit')   
    choice = int(input('Enter your choice: '))

    if choice == 1:
        print('Information of Dataset: \n', df.info)
        print('Shape of Dataset (row x column): ', df.shape)
        print('Columns Name:', df.columns)
        print('Total elements in dataset:', df.size)
        print('Datatype of attributes (columns): ', df.dtypes)
        print('First 5 rows:\n', df.head().T)
        print('Last 5 rows: \n', df.tail().T)   
        print('Any 5 rows: \n',df.sample(5).T)

    if choice == 2:
        print('Statistical information of Numerical Columns: \n', df.describe())

    if choice == 3:
        print('Total Number of Null Values in Dataset:\n', df.isna().sum())

    if choice == 4:
        numcolumns = ['raisedhands', 'VisITedResources', 'AnnouncementsView','Discussion']

        fig, axes = plt.subplots(2,2)
        fig.suptitle('Before removing Outliers')
        sns.boxplot(data = df, x ='raisedhands', ax=axes[0,0])
        sns.boxplot(data = df, x = 'VisITedResources', ax=axes[0,1]) 
        sns.boxplot(data = df, x = 'AnnouncementsView', ax=axes[1,0])
        sns.boxplot(data = df, x = 'Discussion', ax=axes[1,1])
        fig.tight_layout()
        plt.show()

        print('Identifying overall outliers in feature variables.....')
        for var in numcolumns:
            df = DetectOutlier(df, var)

        fig, axes = plt.subplots(2,2)
        fig.suptitle('After removing Outliers')
        sns.boxplot(data = df, x ='raisedhands', ax=axes[0,0])
        sns.boxplot(data = df, x = 'VisITedResources', ax=axes [0,1])
        sns.boxplot(data = df, x = 'AnnouncementsView', ax=axes[1,0])
        sns.boxplot(data = df, x = 'Discussion', ax=axes[1,1])
        fig.tight_layout()
        plt.show()

    if choice == 5:
        df['gender']=df['gender'].astype('category')
        print('Data types of Gender=', df.dtypes['gender'])
        df['gender']=df['gender'].cat.codes
        print('Data types of gender after label encoding = ', df.dtypes['gender'])
        print('Gender Values: ', df['gender'].unique())

    if choice == 6:
        sns.boxplot(data = df, x = 'raisedhands', y='gender', hue = 'gender') 
        plt.title('Boxplot with 2 variables gender and raisedhands')
        plt.show()

    if choice == 7:
        sns.boxplot(data = df, x = 'Discussion', y='NationalITy', hue = 'gender')
        plt.title('Boxplot with 3 variables gender, nationality, discussion') 
        plt.show()

    if choice == 8: 
        sns.scatterplot(data=df, x="raisedhands", y="VisITedResources",hue='gender')
        plt.title('Scatterplot for raisedhands, VisITedResources')
        plt.show()

    if choice == 9:
        break
