import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
def DetectOutlier(df, var):
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    high, low = Q3+1.5*IQR, Q1-1.5*IQR

    print("Highest allowed in variable:", var, high)
    print("Lowest allowed in variable:", var, low)
    count = df[(df[var] > high) | (df[var] < low)][var].count()
    print('Total outliers in:', var, ':', count)

    df = df[((df[var] >= low) & (df[var] <= high))]
    print('	Outliers removed in', var)
    return df


def DrawBoxPlot(df, msg):
    fig, axes = plt.subplots(1, 2)
    fig.suptitle(msg)
    sns.boxplot(data=df, x='rm', ax=axes[0])
    sns.boxplot(data=df, x='lstat', ax=axes[1])
    fig.tight_layout()
    plt.show()


df = pd.read_csv('BostonHousing.csv')
print("Boston housing dataset is successfully loaded")
choice = 1
while (choice != 9):
    print("----------MENU--------")
    print("1. Display information od dataset")
    print("2. Find missing values")
    print("3. Detect and remove outliers")
    print("4. Encoding using label encoder")
    print("5. Find correlation matrix")
    print("6. Train and test model using linear regression model")
    print("7. predict housing price by giving user input")
    print("8. reloaded boston housing dataset")
    print("9. Exit")
    choice = int(input("Enter your choice: "))

    if (choice == 1):
        print(df.head().T)
        print(df.columns)

    if (choice == 2):
        print(df.isnull().sum())

    if (choice == 3):
        Column_Name = ['lstat', 'rm']
        Output = ['medv']

        DrawBoxPlot(df, 'Before removing outliers')
        print('Identifying overall outliers in column name variable')
        for var in Column_Name:
            df = DetectOutlier(df, var)

        df = DetectOutlier(df, 'rm')
        DrawBoxPlot(df, 'After removing outliers')

    if (choice == 4):
        df['medv'] = df['medv'].astype('category')
        print(df.dtypes)
        df['medv'] = df['medv'].cat.codes
        print(df)
        print(df.isnull().sum())

    if (choice == 5):
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.heatmap(df.corr(), annot=True)
        plt.show()

    if (choice == 6):
		
        X = df[['rm', 'lstat']]
        y = df['medv']
        
        X_train, X_test, y_test, y_train = train_test_split(X, y, test_size=0.20, random_state=0)
        print('X_train = ', X_train)
        print('X_test = ', X_test)
		
        model = LinearRegression().fit(np.array(X_train).reshape(-1, 1), y_train)
        y_pred = model.predict(X_test)
        print('y_pred = ', y_pred)
        print('y_test = ', y_test)

        from sklearn.metrics import mean_squared_error

        print('MAE: ', mean_absolute_error(y_test, y_pred))
        print("model score: ", model.score(X_test, y_test))

    if (choice == 7):
        import numpy as np
        from sklearn.linear_model import LinearRegression
        X = df[['rm', 'lstat']]

        y = df['medv']
        
        X_train, X_test, y_test, y_train = train_test_split(X, y, test_size=0.20, random_state=0)
		
        print('X_train = ', X_train)
        X = X.dropna()
        X_train=X_train.dropna()
        model = LinearRegression()
        model.fit((X_train), y_train)

        features = np.array([[6, 19]])
        prediction = model.predict(features)
        print('Prediction: {}'.format(prediction))

    if (choice == 8):
        df = pd.read_csv("BostonHousing.csv")
        print("successfully reloaded boston housing dataset")
        print("Boston hosing dataset \n ", df)

    if (choice == 9):
        break
