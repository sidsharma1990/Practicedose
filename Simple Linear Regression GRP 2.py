# Simple Linear Regression
# Linear = relation between X and y
# regression = 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_excel('Simple Linear Regression.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

###### Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Linear regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

# prediction
reg.predict([[3.6]])
pred = reg.predict(X_test)

# Visualization
plt.scatter(X_train, y_train, color = 'g')
plt.plot (X_train, reg.predict(X_train), color = 'red')
plt.title('Simple Linear Regression')
plt.xlabel('EXP')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, color = 'g')
plt.plot (X_test, reg.predict(X_test), color = 'red')
plt.title('Simple Linear Regression')
plt.xlabel('EXP')
plt.ylabel('Salary')
plt.show()

################### to save model
# Pickle
import pickle
with open('model', 'wb') as file:
    pickle.dump(reg, file)

with open('model', 'rb') as file:
    sm = pickle.load(file)

sm.predict([[3.6]])

###################################
















