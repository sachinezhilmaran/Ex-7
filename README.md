# Ex-07-Feature-Selection

## AIM:

To Perform the various feature selection techniques on a dataset and save the data to a file. 

## EXPLANATION:

Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

## ALGORITHM:

### STEP 1;
Read the given Data
### STEP 2;
Clean the Data Set using Data Cleaning Process
### STEP 3;
Apply Feature selection techniques to all the features of the data set
### STEP 4;
Save the data to the file


## CODE:

NAME: sivakumar A
REG NO: 212220043004
```
from sklearn.datasets import load_boston
boston_data=load_boston()
import pandas as pd
boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
boston['MEDV'] = boston_data.target
dummies = pd.get_dummies(boston.RAD)
boston = boston.drop(columns='RAD').merge(dummies,left_index=True,right_index=True)
X = boston.drop(columns='MEDV')
y = boston.MEDV
boston.head(10)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from math import sqrt

cv = KFold(n_splits=10, random_state=None, shuffle=False)
classifier_pipeline = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10))
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),2)))
print("R_squared: " + str(round(r2_score(y,y_pred),2)))

boston.var()

X = X.drop(columns = ['NOX','CHAS'])
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),2)))
print("R_squared: " + str(round(r2_score(y,y_pred),2)))
```
```
### Filter Features by Correlation

import seaborn as sn
import matplotlib.pyplot as plt
fig_dims = (12, 8)
fig, ax = plt.subplots(figsize=fig_dims)
sn.heatmap(boston.corr(), ax=ax)
plt.show()
abs(boston.corr()["MEDV"])
abs(boston.corr()["MEDV"][abs(boston.corr()["MEDV"])>0.5].drop('MEDV')).index.tolist()
vals = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
for val in vals:
    features = abs(boston.corr()["MEDV"][abs(boston.corr()["MEDV"])>val].drop('MEDV')).index.tolist()
    
    X = boston.drop(columns='MEDV')
    X=X[features]
    
    print(features)

    y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
    print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),2)))
    print("R_squared: " + str(round(r2_score(y,y_pred),2)))
```
```
### Feature Selection Using a Wrapper

boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
boston['MEDV'] = boston_data.target
boston['RAD'] = boston['RAD'].astype('category')
dummies = pd.get_dummies(boston.RAD)
boston = boston.drop(columns='RAD').merge(dummies,left_index=True,right_index=True)
X = boston.drop(columns='MEDV')
y = boston.MEDV

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

sfs1 = SFS(classifier_pipeline, 
           k_features=1, 
           forward=False, 
           scoring='neg_mean_squared_error',
           cv=cv)

X = boston.drop(columns='MEDV')
sfs1.fit(X,y)
sfs1.subsets_

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT']]
y = boston['MEDV']
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))
print("R_squared: " + str(round(r2_score(y,y_pred),3)))

boston[['CRIM','RM','PTRATIO','LSTAT','MEDV']].corr()

boston['RM*LSTAT']=boston['RM']*boston['LSTAT']

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT']]
y = boston['MEDV']
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))
print("R_squared: " + str(round(r2_score(y,y_pred),3)))

sn.pairplot(boston[['CRIM','RM','PTRATIO','LSTAT','MEDV']])

boston = boston.drop(boston[boston['MEDV']==boston['MEDV'].max()].index.tolist())

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT','RM*LSTAT']]
y = boston['MEDV']
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))
print("R_squared: " + str(round(r2_score(y,y_pred),3)))

boston['LSTAT_2']=boston['LSTAT']**2

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT']]
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))
print("R_squared: " + str(round(r2_score(y,y_pred),3)))
```
## OUPUT:

### Read the given dataset;
![image](https://user-images.githubusercontent.com/119560261/235057764-94e583fe-6963-4135-96dc-e938835b451d.png)

### Finding of Errors;
![image](https://user-images.githubusercontent.com/119560261/235057903-90fd0be0-39bc-4022-972c-a399af5dc275.png)
![image](https://user-images.githubusercontent.com/119560261/235057929-7c937252-12b9-4b57-a164-7bc407f8e3d7.png)
![image](https://user-images.githubusercontent.com/119560261/235057966-ed122e0e-243e-4c5d-97d4-f60878160681.png)

### Filter Features by Correlation; 2
![image](https://user-images.githubusercontent.com/119560261/235058213-ed18a2af-ef57-4f14-9d97-3f9a60ce8272.png)

### Feature Selection Using a Wrapper; 4
![image](https://user-images.githubusercontent.com/119560261/235058321-b7a8a001-9890-4dca-b460-87fe4e1b4899.png)
![image](https://user-images.githubusercontent.com/119560261/235058351-f385f5f3-128f-448a-99ae-916802659595.png)
![image](https://user-images.githubusercontent.com/119560261/235058390-0d197ce7-fef0-45a5-8741-c483d86d7d30.png)
![image](https://user-images.githubusercontent.com/119560261/235058438-1a790247-2b67-4495-98af-9494b3279d0f.png)
![image](https://user-images.githubusercontent.com/119560261/235058459-48bacb81-4769-44f2-83e0-15ee62c37ae6.png)
![image](https://user-images.githubusercontent.com/119560261/235058508-a8551d39-7af6-4132-aed2-26d3dffd2bae.png)

### Pair plot;
![image](https://user-images.githubusercontent.com/119560261/235058619-17d4f539-7a9d-4c1a-af0c-d5a4370d481d.png)

## RESULT:
The various feature selection techniques has been performed on a dataset.
