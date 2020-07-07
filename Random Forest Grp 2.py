# Random Forest (Esemble) - Bootstrap Aggregation - Bagging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=10, random_state = 0)
reg.fit(X,y)

reg.predict([[6.5]])

plt.scatter(X,y)
plt.plot(X, reg.predict(X), color = 'g')
plt.title('RF')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()













