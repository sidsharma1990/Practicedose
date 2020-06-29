# Multi Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_excel('Multi Linear regression House price.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

dataset.bedrooms.median()
dataset['bedrooms'].median()

dataset.bedrooms.fillna(dataset.bedrooms.median(), inplace = True)

# Linear regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(dataset[['area', 'bedrooms', 'age']], dataset.price)

# Prediction
reg.predict([[3100, 4, 4]])

# Cofficient
reg.coef_

# intercept/Slope/ Constant
reg.intercept_


y = b0+b1x1+b2x2+......+bnxn

550000 = 221323+ ((2600*112)+(23388*3)+(20*-3231))

