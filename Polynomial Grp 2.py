# Polynomial Grp 2

# Underfitting : - Training and testing both are giving error, but in testing error rate is low than training
# overfitting : - Trainin data gives low error, test data gives high error

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_excel('Age polynomial regression.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

###### Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Linear regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

# Prediction
reg.predict(X_test)

# Visualizing Linear regression
plt.scatter(X_train, y_train, color = 'g')
plt.plot (X_train, reg.predict(X_train), color = 'red')
plt.title('Simple Linear Regression')
plt.xlabel('EXP')
plt.ylabel('Salary')
plt.show()

# Polynomial feature
from sklearn.preprocessing import PolynomialFeatures
poly1 = PolynomialFeatures(degree = 3)
X_poly = poly1.fit_transform(X_train)

# integrated polynomial and linear regression
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y_train)

# normal prediciton
reg.predict([[55]])

poly_reg.predict(poly1.fit_transform([[55]]))
poly_reg.predict(poly1.fit_transform(X_test))

# Visualizing polynomial training regression
plt.scatter(X_train, y_train, color = 'g')
plt.plot (X_train, poly_reg.predict(poly1.fit_transform(X_train)), 
          color = 'red')
plt.title('Polynomial Regression')
plt.xlabel('EXP')
plt.ylabel('Salary')
plt.show()


plt.scatter(X_test, y_test, color = 'g')
plt.plot (X_test, poly_reg.predict(poly1.fit_transform(X_test)), 
          color = 'red')
plt.title('Polynomial Regression')
plt.xlabel('EXP')
plt.ylabel('Salary')
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.loc[:, 'Level'].values
y = dataset.loc[:, 'Salary'].values
X = X.reshape(len(X),1)
# y = y.reshape(len(X),1)


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

lin_reg = LinearRegression()
lin_reg.fit(X,y)

poly_reg = PolynomialFeatures (degree = 1)
X_poly = poly_reg.fit_transform(X)

# Integrate
lin_reg_2  = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Linear Regression
# lin_reg.predict([[6.5]])
plt.scatter(X, y, color = 'g')
plt.plot (X, lin_reg.predict(X), color = 'red')
plt.title('Linear Regression')
plt.xlabel('EXP')
plt.ylabel('Salary')
plt.show()

# Polynomial Regression
# lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
plt.scatter(X, y, color = 'g')
plt.plot (X, lin_reg_2.predict(poly_reg.fit_transform(X)), 
          color = 'red')
plt.title('Polynomial Regression')
plt.xlabel('EXP')
plt.ylabel('Salary')
plt.show()








