# SVR (Support Vector Regression)

# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# imporrting the data
dataset = pd.read_csv('Position_Salaries.csv')

# dividing the data into independent and dependent variable
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y),1)

# Standardization
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


from sklearn.svm import SVR
reg = SVR(kernel='rbf')
reg.fit(X,y)

# prediction
reg.predict([[6.5]])
reg.predict(sc_X.transform([[6.5]]))

# Inverse Transform
sc_y.inverse_transform(reg.predict(sc_X.transform([[15]])))
sc_y.inverse_transform(reg.predict(sc_X.transform([[9.5]])))

# Visualization
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y))
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(reg.predict(sc_X.transform(X))))
plt.title('SVR')
plt.xlabel('Level')
plt.ylabel ('Salary')















