
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder
import pickle

df =pd.read_csv("autos.csv",header=0,sep=',',encoding='Latin1',)



df.seller.value_counts()

df[df.seller != 'gewerblich']

df=df.drop('seller',1)

df.offerType.value_counts()

df[df.offerType !='Gesuch']

df=df.drop('offerType',1)

df.shape

df.powerPS.value_counts()

df=df[(df.powerPS > 50)&(df.powerPS <900)]
df.shape

df.yearOfRegistration.value_counts()

df=df[(df.yearOfRegistration >=1950)&(df.yearOfRegistration <2017)]
df.shape

df.drop(['name','abtest','dateCrawled','nrOfPictures','lastSeen','postalCode','dateCreated'],axis='columns',inplace=True)

new_df =df.copy()
new_df =new_df.drop_duplicates(['price','vehicleType','yearOfRegistration','gearbox','powerPS','model','kilometer','monthOfRegistration','fuelType','notRepairedDamage'])

new_df.gearbox.replace(('manuell','automatik'),('manual','automatic'),inplace=True)
new_df.fuelType.replace(('benzin','andere','elektro'),('petrol','others','electric'),inplace=True)
new_df.vehicleType.replace(('kleinwagen','cabria','kombi','andere'),('small car','convertible','combination','others'),inplace=True)
new_df.notRepairedDamage.replace(('ja','nein'),('Yes','No'),inplace=True)

df.price.value_counts()

df.shape

df.vehicleType.value_counts()

new_df =new_df[(new_df.price >=100)&(new_df.price <=150000)]

df.fuelType.value_counts()

df.gearbox.value_counts()

df.isnull()

df['fuelType'].isnull().any()

df['gearbox'].isnull().any()

df['model'].isnull().any()

df['notRepairedDamage'].isnull().any()

df['vehicleType'].isnull().any()

df['kilometer'].isnull().any()

df['yearOfRegistration'].isnull().any()

df['monthOfRegistration'].isnull().any()

df['powerPS'].isnull().any()

df['price'].isnull().any()

import seaborn as sns
sns.heatmap(df.corr(),annot=True)

sns.pairplot(df)

new_df['notRepairedDamage'].fillna(value='not-declared',inplace=True)
new_df['fuelType'].fillna(value='not-declared',inplace=True)
new_df['gearbox'].fillna(value='not-declared',inplace=True)
new_df['vehicleType'].fillna(value='not-declared',inplace=True)
new_df['model'].fillna(value='not-declared',inplace=True)

sns.heatmap(new_df.corr(),annot=True)

sns.pairplot(new_df)

new_df.to_csv("autos_prep.csv")

x=new_df['price']
x

y=new_df



























