# Data Preprocessing Template Grp 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.loc[:, 'Purchased'].values

# through Fit and transform seprately
from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values=np.nan,
                       strategy="median")
impute.fit(X[:,1:3])
X[:, 1:3] = impute.transform (X[:, 1:3])

# Encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer (transformers = [('encode', OneHotEncoder(),[0])],
                        remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

### Dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

###### Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2,
                                                    random_state = 42)















