##Importing required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
import pickle

#Importing(reading) the dataset
auto = pd.read_csv('https://raw.githubusercontent.com/harshlangade19/Data_Analysis_MS_Engage_2022/main/Car_sales.csv')
auto.head()

auto.describe()

auto.info()

auto.isnull().sum()

#Defining a function for dealing with missing values(null values)
def missing_value(var, stats = 'mean'):
    if (var.dtypes == 'float64') | (var.dtypes == 'int64'):
        var = var.fillna(var.mean()) if stats == 'mean' else var.fillna(var.median())
    else:
        var = var.fillna(var.mode())
    return var

#Seperating continuous and categorical features in the dataset and deal with missing values
continuous_vars = auto.loc[:, (auto.dtypes == 'float64') | (auto.dtypes == 'int64')]
continuous_vars = continuous_vars.apply(lambda x: x.clip(lower = x.quantile(0.01), upper = x.quantile(0.99)))
continuous_vars = continuous_vars.apply(missing_value)

auto['Vehicle_type'].isna().sum()

#Label Encoding 'Vehicle_type' feature to convert it into numeric data
le = LabelEncoder()
auto['Vehicle_type'] = le.fit_transform(auto['Vehicle_type'])

#Creating final dataset by concatenation
auto_new = pd.concat([continuous_vars, auto['Vehicle_type']], axis = 1)
auto_new.head()

#Dropping unneccessary feature columns from the dataset
auto_new.drop(['__year_resale_value', 'Power_perf_factor'], axis = 1, inplace = True)

#Seperating input features(for model) and label column
X = auto_new.drop(['Sales_in_thousands'],axis=1)
y = auto_new['Sales_in_thousands']
X = np.asarray(X).astype(np.float32)
y = np.asarray(y).astype(np.float32)

#Seperating the data into training data and test data
X_train, X_cv, y_train, y_cv = train_test_split(X,y, test_size = 0.3, random_state = 42)

#Implementing feature scaling on the data
scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.fit_transform(X_cv)


#Building ml models for prediciton and choosing the best one 
#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()
#from sklearn.ensemble import RandomForestRegressor
#regressor = RandomForestRegressor(n_estimators=50,random_state=10)
#from sklearn.svm import SVR
#regressor = SVR(kernel='rbf')
from sklearn.linear_model import Lasso
regressor = Lasso(alpha=0.5)

#Fitting the model on training data
regressor.fit(X_train,y_train)

#Predicting the required output using test data
predictions = regressor.predict(X_cv)
predictions

#Evaluating error metrics
mae = mean_absolute_error(y_cv,predictions)
mse = mean_squared_error(y_cv,predictions)
rmse = np.sqrt(mse)
print("Mean Absolute Error = ",mae)
print("Mean Squared Error = ",mse)
print("Root Mean Squared Error = ",rmse)

#Creating a new,random input set
x = np.array([[23.98,1.8,150,102.8,68,178,2.99,16.5,27.1,1]])
x = x.astype(float)
x

#Predicting output for above defined input
y = regressor.predict(x)
y

# Saving model to disk
pickle.dump(regressor, open('model_sales.pkl','wb'))
 
# Loading model to compare the results
model = pickle.load(open('model_sales.pkl','rb'))
print(model.predict([[23.98,1.8,150,102.8,68,178,2.99,16.5,27.1,1]])) 