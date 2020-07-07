# Decision Tree Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor()
reg.fit(X,y)

reg.predict([[14]])

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, 
#                                                     test_size = 0.2,
#                                                     random_state = 0)

# from sklearn.tree import DecisionTreeRegressor
# reg = DecisionTreeRegressor()
# reg.fit(X_train, y_train)

# pred = reg.predict(X_test)

# Score
# from sklearn.metrics import r2_score
# r2_score(y_test, pred)

# Normal Visualization
plt.scatter(X,y)
plt.plot(X, reg.predict(X), color = 'g')
plt.title('DT')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
############################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor()
reg.fit(X,y)


X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X,y)
plt.plot(X_grid, reg.predict(X_grid), color = 'g')
plt.title('DT')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()


















