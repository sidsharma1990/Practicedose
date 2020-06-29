# Polynomial Reg testing 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Regression - Data.csv')
X = dataset.iloc[:, [0,1,3,4]].values
y = dataset.iloc[:, 2].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2,
                                                    random_state = 0)

# Linear regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

# Polynomial feature
from sklearn.preprocessing import PolynomialFeatures
poly1 = PolynomialFeatures()
X_poly = poly1.fit_transform(X_train)

# integrated polynomial and linear regression
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y_train)

y_pred = poly_reg.predict(poly1.fit_transform(X_test))

# Score
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

37.47%









