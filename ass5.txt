def DetectOutlier(df,var): 
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    high, low = Q3+1.5*IQR, Q1-1.5*IQR
    
    print("Highest allowed in variable:", var, high)
    print("lowest allowed in variable:", var, low)
    count = df[(df[var] > high) | (df[var] < low)][var].count()
    print('Total outliers in:',var,':',count)

    df = df[((df[var] >= low) & (df[var] <= high))]
    print('Outliers removed in', var)
    return df

def Display(y_test,y_pred):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print('confusion_matrix\n',cm)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm,annot=True,linewidths=.3,cmap="Blues")
    plt.show()
    import warnings
    warnings.filterwarnings('always')
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))

def DrawBoxplot(df, msg):
    fig, axes = plt.subplots(figsize =(10, 7),ncols=2)
    fig.suptitle(msg)
    sns.boxplot(data = df, x ='Age', ax=axes[0])
    sns.boxplot(data = df, x ='EstimatedSalary', ax=axes[1])
    fig.tight_layout()
    plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("Social_Network_Ads.csv")
print('Social Network Ads dataset is successfully loaded .....')
choice = 1
while(choice != 10):
    print('-------------- Menu --------------------')
    print('1. Display information of dataset')
    print('2. Find Missing values')
    print('3. Detect and remove outliers')
    print('4. Encoding using label encoder')
    print('5. Find correlation matrix')
    print('6. Train and Test the model and Apply Logistic Regration Model')
    print('7. Normlize Data and Apply Logistic Regration Model')
    print('8. Predict Purchased or not by giving User Input ')
    print('9. Reload Social Network Ads dataset successfully')
    print('10. Exit')
    choice = int(input('Enter your choice: '))

    if choice == 1:
        print(df.head().T)
        print(df.columns)
        df.drop(columns=['User ID'],inplace=True)
        df.sample(10)

    if choice == 2:
        print(df.isnull().sum())

    if choice == 3:
        Column_Name = ['Age','EstimatedSalary']
        Output = ['Purchased']

        DrawBoxplot(df, 'Before removing Outliers')
        print('Identifying overall outliers in Column Name variables.....')
        for var in Column_Name:
            df = DetectOutlier(df,var)
        DrawBoxplot(df, 'After removing Outliers')
    
    if choice == 4:
        df['Gender']=df['Gender'].astype('category')
        print(df.dtypes)
        df['Gender']=df['Gender'].cat.codes
        print(df)
        print(df.isnull().sum())

    if choice == 5:
        import seaborn as sns
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(df.corr(),annot=True,linewidths=.3)
        plt.show()

    if choice == 6:
        x = df[['Age','EstimatedSalary']]
        y = df['Purchased']
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test =train_test_split(x,y,test_size= 0.20, random_state=0)       
        print('X_train=',x_train)
        print('X_test=',x_test)

        from sklearn.linear_model import LogisticRegression
        model=LogisticRegression(random_state = 0, solver='lbfgs')
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)

        print('y_test :' ,y_test)
        print('y_pred :' ,y_pred)
        
        print ('Model Score:',model.score(x_test,y_test))

    if choice == 7:
        x = df[['Age','EstimatedSalary']]
        y = df['Purchased']
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test =train_test_split(x,y,test_size= 0.20, random_state=0)

        from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        x_train = sc_X.fit_transform(x_train)
        x_test = sc_X.fit_transform(x_test) 

        from sklearn.linear_model import LogisticRegression
        model1=LogisticRegression(random_state = 42, solver='lbfgs')
        model1.fit(x_train,y_train)
        y_pred = model1.predict(x_test)
        print('X_train=',x_train[:8])
        print('X_test=',x_test[:8])
        print('y_train=',y_train[:8])
        print('y_test :' ,y_test[:8])
        print('y_pred :' ,y_pred)
        print ('Model Score:',model1.score(x_test,y_test))
        Display(y_test,y_pred)

    if choice == 8:
        new_input=[[1.92295008,0],[26, 35000],[38,50000],[36,144000],[40,61000]]
        new_output= model1.predict(new_input)
        print(new_input,new_output)


    if choice == 9:
        df=pd.read_csv("Social_Network_Ads.csv")
        print("Successfully Reloaded the Social_Network_Ads Dataset")
        print("Social_Network_Ads Dataset \n",df)

    if choice == 10:
        break