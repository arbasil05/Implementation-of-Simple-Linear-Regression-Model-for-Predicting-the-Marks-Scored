# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### Step 1: Import the standard Libraries.
#### Step 2: Set variables for assigning dataset values.
#### Step 3: Import linear regression from sklearn.
#### Step 4: Assign the points for representing in the graph.
#### Step 5: Predict the regression for marks by using the representation of the graph.
#### Step 6: Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/admin/Downloads/student_scores.csv")
df.head()
df.tail()
#segregating data to variables
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
#splitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
#displaying predicted values
Y_pred
Y_test
#graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(X_test,Y_test,color='black')
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(Y_test,Y_pred)
print('MSE= ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE= ',mae)
rmse=np.sqrt(mse)
print('RMSE= ',rmse)
```

## Output:
### Y_Prediction:
![image](https://github.com/arbasil05/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144218037/fa9aaddc-d1ae-4ef6-b01f-a49438e843a1)
### Output graphs:
![image](https://github.com/arbasil05/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144218037/58d1d5d7-540c-4e17-b720-f3fe414257a2)

![image](https://github.com/arbasil05/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144218037/39c3cb3c-c15b-4bd2-bb7c-07429d29dab0)






## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
