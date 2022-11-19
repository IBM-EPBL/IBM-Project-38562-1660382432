from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder
import pickle

df =pd.read_csv("/content/drive/MyDrive/ibm project/drive/autos.csv",header=0,sep=',',encoding='Latin1',)

"""df =pd.read_csv("autos.csv",header=0,sep=',')

"""

df.seller.value_counts()

df[df.seller != 'gewerblich']

df=df.drop('seller',1)

df.offerType.value_counts()

df[df.offerType !='Gesuch']

df=df.drop('offerType',1)

df.shape

df=df[(df.powerPS > 50)&(df.powerPS <900)]
df.shape

df=df[(df.yearOfRegistration >=1950)&(df.yearOfRegistration <2017)]
df.shape

df.drop(['name','abtest','dateCrawled','nrOfPictures','lastSeen','postalCode','dateCreated'],axis='columns',inplace=True)

new_df =df.copy()
new_df =new_df.drop_duplicates(['price','vehicleType','yearOfRegistration','gearbox','powerPS','model','kilometer','monthOfRegistration','fuelType','notRepairedDamage'])

new_df.gearbox.replace(('manuell','automatik'),('manual','automatic'),inplace=True)
new_df.fuelType.replace(('benzin','andere','elektro'),('petrol','others','electric'),inplace=True)
new_df.vehicleType.replace(('kleinwagen','cabria','kombi','andere'),('small car','convertible','combination','others'),inplace=True)
new_df.notRepairedDamage.replace(('ja','nein'),('Yes','No'),inplace=True)

new_df =new_df[(new_df.price >=100)&(new_df.price <=150000)]

new_df['notRepairedDamage'].fillna(value='not-declared',inplace=True)
new_df['fuelType'].fillna(value='not-declared',inplace=True)
new_df['gearbox'].fillna(value='not-declared',inplace=True)
new_df['vehicleType'].fillna(value='not-declared',inplace=True)
new_df['model'].fillna(value='not-declared',inplace=True)

new_df.to_csv("autos_prep.csv")

labels =['gearbox','notRepairedDamage','model','brand','fuelType','vehicleType']
mapper={}
for i in labels:
    mapper[i]=LabelEncoder()
    mapper[i].fit(new_df[i])
    tr=mapper[i].transform(new_df[i])
    np.save(str('classes'+i+'.npy'),mapper[i].classes_)
    print(i,":",mapper[i])
    new_df.loc[:,i+'_labels']=pd.Series(tr,index=new_df.index)
labeled=new_df[['price','yearOfRegistration','powerPS','kilometer','monthOfRegistration']+[x+"_labels" for x in labels]]
print(labeled.columns)

Y=labeled.iloc[:,0].values
X =labeled.iloc[:,1:].values

Y =Y.reshape(-1,1)

from sklearn.model_selection import cross_val_score,train_test_split
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.3,random_state=3)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

regressor =RandomForestRegressor(n_estimators=1500,max_depth=14,random_state=34)

regressor.fit(X_train, np.ravel(Y_train,order='C'))

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

regressor1 =RandomForestRegressor(n_estimators=1000,max_depth=10,random_state=34)

regressor1.fit(X_train, np.ravel(Y_train,order='C'))

r1_pred =regressor1.predict(X_test)
print(r2_score(Y_test,r1_pred))

r_pred =regressor.predict(X_test)
print(r2_score(Y_test,r_pred))
#nest ="1500" rstate="34"

Y_test.ndim

Y_test

r_pred.ndim

Y_test.resize(83574, refcheck=False)
Y_test.ndim

Y_test.resize(14964,refcheck=False)
len(Y_test)

len(r1_pred)

r1_pred.resize(14964,refcheck=False)

print(len(r_pred),len(Y_test))

profit=pd.DataFrame({'Actual_y_value':Y_test,'Random_forest':r1_pred})
profit

r_pred.resize(14964,refcheck=False)

from sklearn import metrics
metrics.accuracy_score(Y_test,r_pred)

from sklearn import metrics
confusion_matrix =metrics.confusion_matrix(Y_test,r_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix= confusion_matrix,display_labels = [False,True])

r_pred.resize(14964, refcheck=False)

len(Y_test)

profit=pd.DataFrame({'Actual_y_value':Y_test,'Random_forest':r_pred})
profit

from sklearn.metrics import mean_squared_error
print(mean_squared_error(Y_test, r_pred))

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(Y_test,r_pred))

r_pred.resize(14964,refcheck=False)
print(len(Y_test),len(r_pred))

r_pred =regressor.predict(X_test)
print(r2_score(Y_test,r_pred))
#nest ="1000" rstate="34"

r_pred =regressor.predict(X_test)
print(r2_score(Y_test,r_pred))
#nest="15000" rstate="42"

r_pred =regressor.predict(X_test)
print(r2_score(Y_test,r_pred))

from sklearn.linear_model import Ridge

r=Ridge()

r.fit(X_train,Y_train)

ridge_pred=r.predict(X_test)
ridge_pred

ridge_pred_train = r.predict(X_train)
ridge_pred_train

from sklearn.metrics import r2_score


y_pred =r.predict(X_test)
print(r2_score(Y_test,y_pred))



from sklearn.linear_model import Lasso

l=Lasso()

l.fit(X_train,Y_train)

lasso_pred=l.predict(X_test)
lasso_pred

lasso_pred_train = l.predict(X_train)
lasso_pred_train

Y_test

r_pred

ridge_pred.ndim



lasso_pred

Y_test.resize(83574,)
Y_test.ndim

ridge_pred.resize(83574,refcheck=False)

lasso_pred.resize(83574,)

profit=pd.DataFrame({'Actual_y_value':Y_test,'Ridge_pred':ridge_pred,'Lasso_pred':lasso_pred,'Random_forest':r_pred})
profit



from sklearn.metrics import r2_score
from sklearn import metrics 

y_pred =l.predict(X_test)
print(r2_score(Y_test,y_pred))

print(metrics.mean_squared_error(Y_test,ridge_pred))
print(metrics.mean_squared_error(Y_test,lasso_pred))
print(metrics.mean_squared_error(Y_test,r1_pred))
# RMSE

print(np.sqrt(metrics.mean_squared_error(Y_test,ridge_pred)))
print(np.sqrt(metrics.mean_squared_error(Y_test,lasso_pred)))
print(np.sqrt(metrics.mean_squared_error(Y_test,r1_pred)))
## R2-score

print(metrics.r2_score(Y_test,ridge_pred))
print(metrics.r2_score(Y_test,lasso_pred))
print(metrics.r2_score(Y_test,r1_pred))
# R2 score on Training data

print(metrics.r2_score(Y_train,ridge_pred_train))
print(metrics.r2_score(Y_train,lasso_pred_train))
print(metrics.r2_score(Y_train,r1_pred_train))

Y_test

ridge_pred

lasso_pred



from sklearn.linear_model import LinearRegression

MLR= LinearRegression()

MLR.fit(X_train,Y_train)

m_pred=MLR.predict(X_test)
m_pred
m_pred.resize(83574,)
m_pred.ndim
lasso_pred.ndim
r_pred.ndim
ridge_pred.ndim
Y_test.resize(83574,)

profit=pd.DataFrame({'Actual_y_value':Y_test,'Ridge_pred':ridge_pred,'Lasso_pred':lasso_pred,'Random_forest':r_pred,'multiple_linear':m_pred})
profit.head(10)

filename = 'MLR.sav'
pickle.dump(MLR,open(filename,'wb'))

filename = 'ridge.sav'
pickle.dump(r,open(filename,'wb'))

filename = 'lasso.sav'
pickle.dump(l,open(filename,'wb'))

filename = 'resale_model.sav'
pickle.dump(l,open(filename,'wb'))