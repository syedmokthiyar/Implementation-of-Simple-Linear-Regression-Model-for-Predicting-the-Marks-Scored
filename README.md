# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Syed Mokthiyar S.M
RegisterNumber:  212222230156
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error
df = pd.read_csv('/content/ml.csv')
df.head(10)

plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')

x=df.iloc[:,0:-1]
y=df.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(x_train,y_train)

x_train
y_train

lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_train,lr.predict(x_train),color='orange')
lr.coef_
lr.intercept_



```

## Output:
# df.head:
![Screenshot 2024-02-23 101858](https://github.com/syedmokthiyar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118787294/c11dfab9-edd8-432c-87b5-3406c16fb805)
# Graph of plotted data:
![Screenshot 2024-02-23 102201](https://github.com/syedmokthiyar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118787294/4b6f8d9a-ed1e-4b30-930c-cb6454092162)
# Performing Linear Regression:
![Screenshot 2024-02-23 103350](https://github.com/syedmokthiyar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118787294/28dc325e-4427-4d36-969f-9e32016d8a2f)
#  Trained data:
![Screenshot 2024-02-23 105331](https://github.com/syedmokthiyar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118787294/8ae6c37d-d03e-4548-a8a6-8ad70c9910eb)

# Predicting the line of Regression:
![Screenshot 2024-02-23 105109](https://github.com/syedmokthiyar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118787294/5680b391-d747-4b43-a8e0-92507d80df78)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
