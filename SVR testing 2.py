## SVR testing 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Regression - Data.csv')
X = dataset.iloc[:, [0,1,3,4]].values
y = dataset.iloc[:, 2].values
y = y.reshape(len(y),1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2,
                                                    random_state = 0)

# Standardization
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)


from sklearn.svm import SVR
reg = SVR(kernel='rbf')
reg.fit(X_train, y_train)

pred = sc_y.inverse_transform(reg.predict(sc_X.transform(X_test)))
# Score
from sklearn.metrics import r2_score
r2_score(y_test, pred)

41.61
