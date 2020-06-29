# Multi Linear Testing 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Regression - Data.csv')
X = dataset.iloc[:, [0,1,3,4]].values
y = dataset.iloc[:, 2].values

# Correlation
corr1 = dataset.corr()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2,
                                                    random_state = 0)

# Linear Regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

# Prediction
pred = reg.predict(X_test)

from sklearn.metrics import r2_score
r2_score(y_test, pred)

Score - 31.21%


















