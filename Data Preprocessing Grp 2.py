# Data Preprocessing
# fit to train the data
# transform - to transform data
# X is matrix of features, y is dependent variable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
X1 = dataset.iloc[:, :-1]

y = dataset.loc[:, 'Purchased'].values
y1 = dataset.iloc[:, -1].values

### pandas way to fill na values
# dataset['Age'].fillna(value = dataset['Age'].median(), inplace = True)
# dataset.Age = dataset.Age.fillna(dataset.Age.mean())
''' df2 = dataset.fillna({'Age': dataset.Age.mean(),
#                       'Salary':dataset.Salary.median()})'''

#through Scikit learn
# imputer (fit - train, transform = transform)
# through Fit_tranform
from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values=np.nan,
                       strategy="median")
X[:, 1:3] = impute.fit_transform (X[:, 1:3])

# through Fit and transform seprately
from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values=np.nan,
                       strategy="median")
impute.fit(X[:,1:3])
X[:, 1:3] = impute.transform (X[:, 1:3])

##### 
# dummy = pd.get_dummies(dataset['State'])
# X = pd.concat([X, dummy], axis = 1)
# X.drop(['State', 'Mumbai'], axis = 1, inplace = True)

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
 
##### Feature Scaling (Standardization)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

# normalizer
# from sklearn.preprocessing import Normalizer
# norma = Normalizer().fit(X)
















