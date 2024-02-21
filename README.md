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
6.Compare the graphs and hence we obtained the linear regression for the given datas

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KARTHICK K
RegisterNumber:  212222040070
```
```
import pandas as pd
df=pd.read_csv('/content/TABLE - Sheet1.csv')
df.head()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/TABLE - Sheet1.csv')
df.head(10)
plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.xlabel('y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
y
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
plt.xlabel('y')
plt.plot(x_train,lr.predict(x_train),color='red')
```

## Output:
![image](https://github.com/karthick960/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121215938/e32d6f60-e591-472b-85ba-ff6e0c51c07a)
![image](https://github.com/karthick960/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121215938/b5c3540d-1d35-46ad-a939-43a71ff3cf06)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
