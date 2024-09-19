# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Necessary Libraries and Load Data.
2. Split Dataset into Training and Testing Sets.
3. Train the Model Using Stochastic Gradient Descent (SGD).
4. Make Predictions and Evaluate Accuracy.
5. Generate Confusion Matrix.

## Program:
```
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Pradeep Kumar G
RegisterNumber:  212223230150
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#load the iris dataset
iris = load_iris()

#create a pandas dataframe
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target

#print the first 5 values
print(df.head())

#split the data into features (x) and(y)
X=df.drop('target',axis=1)
Y=df['target']

#split the data into training and testing sets
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,random_state=42)

#create an SGD classifier with default parameters
sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)

#train the classifier on thr training data
sgd_clf.fit(X_train,Y_train)

#make predictions on the testing data
y_pred=sgd_clf.predict(X_test)

print(f"Accuracy:{accuracy:.3f}")

#calculate the confusion matrix
cf=confusion_matrix(Y_test, y_pred)
print("Confusion Matrix")
print(cf)
```

## Output:

![image](https://github.com/user-attachments/assets/ecc86a65-5f67-41f1-b800-c5d25e021513)

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
