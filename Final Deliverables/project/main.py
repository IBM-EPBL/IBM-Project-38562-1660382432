from flask import Blueprint, render_template
from flask_login import login_required, current_user
from . import db

import pandas as pd
import numpy as np 
from flask import Flask,render_template,Response,request

import jinja2
import pickle
from sklearn.preprocessing import LabelEncoder
import pickle
import os



main = Blueprint('main', __name__)


filename = 'resale_model.sav'
model_rand = pickle.load(open(filename,'rb'))

brand=np.load('classesbrand.npy',allow_pickle=True)
#print(brand)
fuelType=np.load('classesfuelType.npy',allow_pickle=True)
model=np.load('classesmodel.npy',allow_pickle=True)
vehicleType=np.load('classesvehicleType.npy',allow_pickle=True)
gearbox=np.load('classesgearbox.npy',allow_pickle=True)
RorD=np.load('classesnotRepairedDamage.npy',allow_pickle=True)
    
m=1950
n=2017
year=[]

while(m != n+1):
    year.append(m)
    m=m+1
sm=1
em=12
month=[]
while(sm != em+1):
    month.append(sm)
    sm=sm+1

print(model_rand)
@main.route('/get_details',methods=['GET','POST'])
def get_details():
    regno=request.form['tn']
    
    #query to get value from database
    #data = pd.read_csv("autos_prep.csv")
    #data = data[data['regno']==regno]
    
    brand = "bmw"
    year = "2011" 
    #powerps = 1#float(request.form['powerps'])
    #kms = #float(request.form['kms'])
    month = "6"#int(request.form['regmonth'])
    gearbox = "manual"#request.form['gearbox']
    damage = "yes"#request.form['dam']
    model ="200"#request.form.get('modeltype')
    brand ="bmw" #request.form.get('brand')
    fuelType = "cng" #request.form.get('fuel')
    vehicletype = "bus"#request.form.get('vehicletype')
    RorD = "Yes"
    #print(regno)
    #regno1="TN72AU4321"
    
    
    
    return render_template('index.html',brandv=brand,yearv= year,monthv=month,modelv=model,fuelTpv=fuelType,vehicletpv=vehicletype,gearboxv=gearbox,RorDv=RorD)
@main.route('/')
def index():
    brand=np.load('classesbrand.npy',allow_pickle=True)
    #print(brand)
    fuelType=np.load('classesfuelType.npy',allow_pickle=True)
    model=np.load('classesmodel.npy',allow_pickle=True)
    vehicleType=np.load('classesvehicleType.npy',allow_pickle=True)
    gearbox=np.load('classesgearbox.npy',allow_pickle=True)
    RorD=np.load('classesnotRepairedDamage.npy',allow_pickle=True)
    
    m=1950
    n=2017
    year=[]

    while(m != n+1):
       year.append(m)
       m=m+1
    sm=1
    em=12
    month=[]
    while(sm != em+1):
        month.append(sm)
        sm=sm+1
    #  print("hi")
    br="Select your car brand"
    y="Select your car year of Registration"
    md="Select your car model"
    fl="Select your car FuelType"
    mr="Select your car month of Registration"
    gb="Select your car gearbox"
    vp="Select your car VehicleType"
   #brand=brand,year= year,month=month,model=model,RorD=RorD,fuelTp=fuelType,vehicletp=vehicleType,gearbox=gearbox
    return render_template('index.html',brand=brand,year= year,month=month,model=model,RorD=RorD,fuelTp=fuelType,vehicletp=vehicleType,gearbox=gearbox,vehicletpv=vp,gearboxv=gb,monthv=mr,yearv=y,brandv=br,modelv=md,fulev=fl)


@main.route('/y_predict',methods=['GET','POST'])
def y_predict():
    regyear =int(request.form['regyear'])
    powerps = float(request.form['powerps'])
    kms = float(request.form['kms'])
    regmonth = int(request.form['regmonth'])
    gearbox = request.form['gearbox']
    damage = request.form['dam']
    model =request.form.get('modeltype')
    brand = request.form.get('brand')
    fuelType = request.form.get('fuel')
    vehicletype = request.form.get('vehicletype')
    new_row ={'yearOfRegistration':regyear,'powerPS':powerps,'kilometer':kms,'monthOfRegistration':regmonth,'gearbox':gearbox,'notRepairedDamage':damage,'model':model,'brand':brand,'fuelType':fuelType,'vehicleType':vehicletype}
    print(new_row)
    new_df = pd.DataFrame(columns =['yearOfRegistration','powerPS','kilometer','monthOfRegistration','gearbox','notRepairedDamage','model','brand','fuelType','vehicleType'])
    new_df = new_df.append(new_row,ignore_index=True)
    labels = ['gearbox','notRepairedDamage','model','brand','fuelType','vehicleType']

    mapper ={}
    for i in labels:
        mapper[i]=LabelEncoder()
        mapper[i].classes_=np.load(str('classes'+i+'.npy'),allow_pickle=True)
        tr=mapper[i].fit_transform(new_df[i])
        new_df.loc[:,i+'_labels']=pd.Series(tr,index=new_df.index)
    labeled=new_df[['yearOfRegistration','powerPS','kilometer','monthOfRegistration']+[x+'_labels' for x in labels]]
    X=labeled.values
    print(X)

    y_prediction=model_rand.predict(X)
    print(y_prediction)
    yy=y_prediction[0]*80
    print(yy)
    return render_template('index.html',ypred='Resale Price of the car is ₹{:.2f} '.format(yy))



@main.route('/profile')
@login_required
def profile():
    return render_template('profile.html', name=current_user.name)
    
if __name__ == '__main__':
    app.run(host='localhost',debug=True,threaded=False)
