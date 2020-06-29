# Multi Linear Regression GRP 2 pending
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer (transformers = [('encode', OneHotEncoder(),[3])],
                        remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

# Splitting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
 
# Linear Regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

# Prediction
pred = reg.predict(X_test)

######### eliminiation
import statsmodels.api as sm

X1 = np.array(X[:, [0,1,2,3,4,5]], dtype = float)
ols1 = sm.OLS (endog = y, exog = X1).fit()
ols1.summary()

X1 = np.array(X[:, [0,1,2,3,5]], dtype = float)
ols1 = sm.OLS (endog = y, exog = X1).fit()
ols1.summary()

X1 = np.array(X[:, [0,1,2,3]], dtype = float)
ols1 = sm.OLS (endog = y, exog = X1).fit()
ols1.summary()

X1 = np.array(X[:, [3]], dtype = float)
ols1 = sm.OLS (endog = y, exog = X1).fit()
ols1.summary()

# y = mean, y1, y2.....yn = actual values
# OLS = [(y1-y)**2 + (y2-y)**2 + (y3-y)**2+.......+ (yn-y)**2]
# MSE (Mean Sq error) = Cost function

########### R squared, Adjusted R square and Gradeint Descent
# Gradient Descent

# Learning rate






















