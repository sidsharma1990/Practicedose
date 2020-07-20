# Logistic Regression
# - Linear Regression of classification

# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_excel('Logistic regression Home.xlsx')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

plt.scatter(X,y)

# Data Spliting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,
                                                    random_state = None)

# X_train1, X_test1= train_test_split(X,test_size = 0.1,random_state = None)
# y_train1, y_test1= train_test_split(y,test_size = 0.1,random_state = None)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

pred = classifier.predict(X_test)
prob = classifier.predict_proba(X_test)
prob_log = classifier.predict_log_proba(X_test)

#########################################
dataset = pd.read_csv('Car Churn.csv')

X = dataset.iloc[:, [1,2]].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,
                                                    random_state = None)

# X_train, X_test= train_test_split(X,test_size = 0.2,random_state = None)
# y_train, y_test= train_test_split(y,test_size = 0.2,random_state = None)

# Standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print (cm)
accuracy_score(y_test, y_pred)`#0.875

[[51  0]
 [29  0]]

[[49  2] = for 0 = 49
 [ 8 21]] = for 1





