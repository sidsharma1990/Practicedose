# Simple Linear Regression for Grp 1

# x is an independent variable
# y is a dependent variable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.linear_model import LinearRegression

df = pd.read_excel('C:\\Users\\DELL\\Desktop\\Python DS\\Simple Linear regression House price.xlsx')

# labeling
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area, df.price, color = 'red')

####
price = df.price
print (price)

###
# area = df.area
# print (area)
df2 = df.drop('price', axis = 1)

##################
reg = linear_model.LinearRegression()
reg.fit(df2, price)

reg.predict([[3300]])

reg.predict([[8200]])

reg.coef_
reg.intercept_


spliting the data (80-20)
feature selection


y = b0 +b1x1















