from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
iris=load_iris()

ix_train,ix_test,iy_train,iy_test=train_test_split(iris.data,iris.target,random_state=0,test_size=0.2)

iReg=LinearRegression()
iReg.fit(ix_train,iy_train)

yPred=iReg.predict(ix_test)

d=input("enter the sepal and petal length and width:")
d=d.split(",")
l=[]
for i in d:
    l.append(float(i))
t = np.array(l).reshape(1,4)
n=iReg.predict(t)
n=np.round(n)

print("your predicted iris category is:",iris.target_names[int(n)])
