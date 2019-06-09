from flask import Flask,render_template,request
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
import numpy as np
import io
import base64


app=Flask(__name__,static_folder='vendor')

@app.route('/')
def index():
    
    return render_template('index.html')
@app.route('/graph.html')
def graph():
    dataset=pd.read_csv('Salary_Data.csv')
    x=dataset.iloc[:,0:1].values
    y=dataset.iloc[:,1:2].values

    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=1/3)
    sReg=LinearRegression()
    sReg.fit(x_train,y_train)
    plt.scatter(x_train,y_train,color='blue')
    plt.scatter(x_test,y_test,color='green')
    plt.plot(x_train,sReg.predict(x_train),'r')
    #plt.plot(x1,y1,'r-o')
    img=io.BytesIO()
    plt.savefig(img,format='png')
    graphUrl=base64.b64encode(img.getvalue()).decode()
    return render_template('graph.html',graphInfo=graphUrl)

@app.route('/check.html',methods=["GET"])
def checki():
    
    return render_template('check.html')


@app.route('/check.html',methods=["POST"])
def check():
    dataset=pd.read_csv('Salary_Data.csv')
    x=dataset.iloc[:,0:1].values
    y=dataset.iloc[:,1:2].values

    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=1/3)
    sReg=LinearRegression()
    sReg.fit(x_train,y_train)
    sal=[]
    #y_pred=sReg.predict(x_test)
    val=[int(request.form['n1'])]
    l=[]
    for i in val:
        l.append(i)
    t = np.array(l).reshape(len(l),1)
    sal=sReg.predict(t)
    sal=sal.astype(int)
    sal=sal.tolist()
    z=sal[0]
    sal=z[0]
    return render_template('check.html',value=int(sal))

if(__name__=='__main__'):
    app.run(debug=True)

