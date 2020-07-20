# KNN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print (cm)
accuracy_score(y_test, y_pred)  # 0.925

[[50  3]
 [ 3 24]]


classifier.predict(sc.transform([[30,76000]]))











