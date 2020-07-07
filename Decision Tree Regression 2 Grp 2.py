# Decision Tree Regression 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_excel('Decision Tree Regression.xlsx')
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0)

from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor()
reg.fit(X_train,y_train)

reg.predict([[42]])

y_pred = reg.predict(X_test)

# Score
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

X_grid = np.arange(min(X_train), max(X_train))
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X_train,y_train)
plt.plot(X_grid, reg.predict(X_grid), color = 'g')
plt.title('DT')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()


















