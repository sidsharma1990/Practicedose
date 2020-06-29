# Decision Tree Regression

import pandas as pd
import numpy as np

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1]
y = dataset.iloc[:, -1]

from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor()
reg.fit(X,y)

reg.predict([[14]])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2,
                                                    random_state = 0)

from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor()
reg.fit(X_train, y_train)

pred = reg.predict(X_test)

# Score
from sklearn.metrics import r2_score
r2_score(y_test, pred)

























