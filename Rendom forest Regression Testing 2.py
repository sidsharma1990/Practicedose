# Rendom forest Regression Testing 2

import pandas as pd
import numpy as np

dataset = pd.read_csv('Regression - Data.csv')
X = dataset.iloc[:, [0,1,3,4]].values
y = dataset.iloc[:, 2].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2,
                                                    random_state = 0)

from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=100, random_state = 0)
reg.fit(X_train,y_train)

pred = reg.predict(X_test)

# Score
from sklearn.metrics import r2_score
r2_score(y_test, pred)

# 78.32, 79.28 (20), 80.58(100)