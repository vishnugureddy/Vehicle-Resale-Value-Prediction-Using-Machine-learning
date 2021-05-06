# -*- coding: utf-8 -*-
"""


@author: G.Vishnu
"""



import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from joblib import load
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
# model = load('rfg.save')
# load the model
model = load(open('vehicleresaleprice.pkl', 'rb'))
# load the scaler
scaler = load(open('scaler.pkl', 'rb'))
trans=load('transform')
trans1=load('transform1')
trans2=load('transform2')
trans3=load('transform3')
trans4=load('transform4')

@app.route('/')
def home():
    return render_template('index.html')


#database={'jithendra':'gitam@123','likitha':'gitam@123','gurav':'gitam@123','vishnu':'gitam@123'}


#@app.route('/login',methods=['POST','GET'])
#def login():
#    un=request.form['username']
#    pwd=request.form['password']
#    if un not in database:
#        return render_template('login.html',info="Invalid user")
#    else:
#        if database[un]!=pwd:
#            return render_template('login.html',info="Invalid password")
#        else:
#            return render_template('index.html',name=un)
        



@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [[x for x in request.form.values()]]
    print(x_test)
    x_test=trans.transform(x_test)
    x_test=x_test[:,1:]
    x_test=trans1.transform(x_test)
    x_test=trans2.transform(x_test)
    x_test=x_test[:,1:]
    x_test=trans3.transform(x_test)
    x_test=x_test[:,1:]
    x_test=trans4.transform(x_test)
    x_test=scaler.transform(x_test)
    print(x_test)
    prediction = model.predict(x_test)
    print(prediction)
    output=prediction[0]

    return render_template('index.html', prediction_text='price = {0:.3f}'.format(output))

'''@app.route('/predict_api',methods=['POST'])
def predict_api():
    #For direct API calls trought request
    data = request.get_json(force=True)
    prediction = model.y_predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)'''

if __name__ == "__main__":
    app.run(debug=True)