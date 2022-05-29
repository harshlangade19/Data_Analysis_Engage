#Importing required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle


#Importing(reading) the dataset
data = pd.read_csv(
    'https://raw.githubusercontent.com/harshlangade19/Data_Analysis_MS_Engage_2022/main/auto-mpg.csv')
data.head()

data.info()

data.describe()
data.isnull().sum()

#Dropping unneccessary feature columns from the dataset
data = data.drop(['car name', 'origin', 'model year'], axis=1)
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')

#Checking correlationn of 'horsepower'with other features
value = data[data['horsepower'].isnull()].index.tolist()

#Determining a factor to fill null values 
hp_avg = data['horsepower'].mean()
disp_avg = data['displacement'].mean()
factor = hp_avg / disp_avg

#Filling null values in 'horsepower' feature column
for i in value:
    data['horsepower'].fillna(value=(data.iloc[i][2])*factor, inplace=True)
    
#Seperating input features(for model) and label column
X = data.drop('mpg', axis=1)
y = data['mpg']
X = np.asarray(X).astype(np.float32)
y = np.asarray(y).astype(np.float32)

#Seperating the data into training data and test data
X_train, X_cv, y_train, y_cv = train_test_split(
    X, y, test_size=0.3, random_state=42)

#Implementing feature scaling on the data
scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.fit_transform(X_cv)

#Building ml models for prediciton and choosing the best one 
#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=50,random_state=10)
#from sklearn.svm import SVR
#regressor = SVR(kernel='rbf')

#Fitting the model on training data
regressor.fit(X_train, y_train)

#Predicting the required output using test data
predictions = regressor.predict(X_cv)
predictions

#Evaluating error metrics
mae = mean_absolute_error(y_cv, predictions)
mse = mean_squared_error(y_cv, predictions)
rmse = np.sqrt(mse)
print("Mean Absolute Error = ", mae)
print("Mean Squared Error = ", mse)
print("Root Mean Squared Error = ", rmse)

#Creating a new,random input set
x = np.array([[8, 340, 145, 3600, 12, ]])
x = x.astype(float)
x

#Predicting output for above defined input
y = regressor.predict(x)
y

# Saving model to disk
pickle.dump(regressor, open('model.pkl', 'wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[8, 340, 145, 3600, 12]]))
