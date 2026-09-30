from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
car=pd.read_csv("clean_car.csv")
model=pickle.load(open("car_pred_model.pkl","rb"))

@app.route('/')
def index():
    names=sorted(car['name'].unique())
    companies=sorted(car['company'].unique())
    year=sorted(car['year'].unique())
    fuelTypes=sorted(car['fuel_type'].unique())


    return render_template("index.html",names=names,companies=companies,year=year,fuelTypes=fuelTypes)
@app.route('/predict',methods=['POST'])
def predict():
    company=request.form.get('company')
    print(company)
    car_model=request.form.get('car_model')
    print(car_model)
    year=int(request.form.get('year'))
    print(year)
    fuel_type=request.form.get('fuel_type')
    print(fuel_type)
    kms_driven=int(request.form.get('kms_driven'))
    print(kms_driven)
    result=model.predict(pd.DataFrame([[car_model,company,year,kms_driven,fuel_type]],columns=['name','company','year','kms_driven','fuel_type']))
    print(result)
    return "car price(approx):" +str(result[0])

    


if __name__ == '__main__':
    app.run(debug=True)