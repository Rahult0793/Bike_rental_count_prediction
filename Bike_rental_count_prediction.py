# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:53:04 2020

@author: hp
"""

# Importing the libraries
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:,2:15].values
y = dataset.iloc[:, 15].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import OneHotEncoder 
  
# creating one hot encoder object with categorical feature 0 
# indicating the first column 
onehotencoder = OneHotEncoder(categorical_features = [0,2,4,6])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_test = onehotencoder.fit_transform(X_test).toarray()

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()

#model fitting - Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train, y_train)
y_pred_lm = regressor.predict(X_test)

#model fitting - Decision tree regression
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0, max_depth = 15)
regressor.fit(X_train,y_train)
y_pred_dt = regressor.predict(X_test)

#model fitting - Random Forest regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(random_state = 0, max_depth = 10, n_estimators = 400)
regressor.fit(X_train,y_train)
y_pred_rf = regressor.predict(X_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#model fitting - SVM Regression
from sklearn.svm import SVR
svr_reg = SVR(kernel = 'linear', C = 1 , epsilon = 0.01)
svr_reg.fit(X_train,y_train)
y_pred_svm = svr_reg.predict(X_test)

#Performance Evaluation
#1) For linear multiple regression : 
#Mean absolute error
from sklearn import metrics
MAE_lm = metrics.mean_absolute_error(y_test,y_pred_lm)
#Mean squared error
MSE_lm = metrics.mean_squared_error(y_test,y_pred_lm)
#RSME
RSME_lm = np.sqrt(MSE_lm)


#2) For decision tree regression
MAE_dt = metrics.mean_absolute_error(y_test,y_pred_dt)
#Mean squared error
MSE_dt = metrics.mean_squared_error(y_test,y_pred_dt)
#RSME
RSME_dt = np.sqrt(MSE_dt)

#3) For random forest regression
MAE_rf = metrics.mean_absolute_error(y_test,y_pred_rf)
#Mean squared error
MSE_rf = metrics.mean_squared_error(y_test,y_pred_rf)
#RSME
RSME_rf = np.sqrt(MSE_rf)

#3) For SVM regression
MAE_svm = metrics.mean_absolute_error(y_test,y_pred_svm)
#Mean squared error
MSE_svm = metrics.mean_squared_error(y_test,y_pred_svm)
#RSME
RSME_svm = np.sqrt(MSE_svm)