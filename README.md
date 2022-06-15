# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.Import the required packages.

2. Read the given dataset and assign x and y array.

3. Split x and y into training and test set.

4. Scale the x variables.

5. Fit the logistic regression for the training set to predict y.

6. Create the confusion matrix and find the accuracy score, recall sensitivity and specificity.

7. Plot the training set results.

## Program:
```python
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Manoj Guna Sundar Tella.
RegisterNumber:  212221240026.
*/
#importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#reading and displaying dataframe
df=pd.read_csv("Social_Network_Ads (1).csv")
df

#assigning x and y and displaying them
x=df.iloc[:,[2,3]].values
y=df.iloc[:,4].values 

#splitting data into train and test
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)

#scaling values and obtaining scaled array of train and test of x
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
xtrain=sc_x.fit_transform(xtrain)
xtest=sc_x.fit_transform(xtest)

#applying logistic regression to the scaled array
from sklearn.linear_model import LogisticRegression
c=LogisticRegression(random_state=0)
c.fit(xtrain,ytrain)

#finding predicted values of y
ypred=c.predict(xtest)
ypred

#calculating confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,ypred)
cm

#calculating accuracy score
from sklearn import metrics
acc=metrics.accuracy_score(ytest,ypred)
acc

#calculating recall sensitivity and specificity
r_sens=metrics.recall_score(ytest,ypred,pos_label=1)
r_spec=metrics.recall_score(ytest,ypred,pos_label=0)
r_sens,r_spec

#displaying regression 
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
xs,ys=xtrain,ytrain
x1,x2=np.meshgrid(np.arange(start=xs[:,0].min()-1,stop=xs[:,0].max()+1,step=0.01),
               np.arange(start=xs[:,1].min()-1,stop=xs[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,c.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
                            alpha=0.75,cmap=ListedColormap(("pink","purple")))
plt.xlim(x1.min(),x2.max())
plt.ylim(x2.min(),x1.max())
for i,j in enumerate(np.unique(ys)):
    plt.scatter(xs[ys==j,0],xs[ys==j,1],
                c=ListedColormap(("white","violet"))(i),label=j)
plt.title("Logistic Regression(Training Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
```

## Output:
![img1](https://user-images.githubusercontent.com/94883876/173859916-ce424505-7457-44f8-a3b9-4c0fa1ca2bee.jpg)
![img2](https://user-images.githubusercontent.com/94883876/173859942-27450cc9-6daa-4e72-ae9e-3ff1afada440.jpg)
![img3](https://user-images.githubusercontent.com/94883876/173859962-dd287c91-9043-410b-9048-82d5784507d1.jpg)
![img4](https://user-images.githubusercontent.com/94883876/173859992-e623ac87-e4fc-41c8-8d3f-7f341b737de0.jpg)
![img5](https://user-images.githubusercontent.com/94883876/173860012-cab7135d-f9c0-43fa-bfb1-53056663361d.jpg)
![img6](https://user-images.githubusercontent.com/94883876/173860029-c2772a26-debf-4326-8d17-56647c5e61a0.jpg)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

