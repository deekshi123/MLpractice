import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 

dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,0:1].values
y=dataset.iloc[:,1:2].values

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=1/3)

sReg=LinearRegression()

sReg.fit(x_train,y_train)

y_pred=sReg.predict(x_test)

plt.subplot(1,2,1)
plt.scatter(x_train,y_train,color='blue')
plt.plot(x_train,sReg.predict(x_train),'r')
plt.title('plotted traing data')

plt.subplot(1,2,2)
plt.scatter(x_test,y_test,color='green')
plt.plot(x_train,sReg.predict(x_train),'b')
plt.title('plotted testing data')

val=input("enter the year of experience:")
val=val.split(',')
l=[]
for i in val:
    print(i)
    l.append(float(i))
t = np.array(l).reshape(len(l),1)
sal=sReg.predict(t)
sal=sal.astype(int)
sal=sal.tolist()
z=sal[0]
sal=z[0]
print("your salary is ",int(sal))
