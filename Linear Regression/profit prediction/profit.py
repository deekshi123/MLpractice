import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dataset=pd.read_csv('database.csv')
x=dataset.iloc[:,:4].values
y=dataset.iloc[:,4:].values

#convert categorical data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

lEncoder=LabelEncoder()
x[:,3]=lEncoder.fit_transform(x[:,3])

ohEncoder=OneHotEncoder(categorical_features=[3])
x=ohEncoder.fit_transform(x).toarray(())
x=x[:,1:]

#split the data
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=1/5)

#apply regression model
Reg=LinearRegression()
Reg.fit(x_train,y_train)

#predict your test case
y_pred=Reg.predict(x_test)


import statsmodels.formula.api as sm

a=np.ones((50,1))
x=np.append(a,x,axis=1)

xopt=x[:,[4]]

#check most preferable labels, then make changes to x value.
sm.OLS(endog=y,exog=xopt).fit().summary()

newReg=LinearRegression()

xop_Train,xop_Test,yop_Train,yop_Test=train_test_split(xopt,y,random_state=1,test_size=0.2)
newReg.fit(xop_Train,yop_Train)
newyPred=newReg.predict(xop_Test)

