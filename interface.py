from flask import Flask,render_template,jsonify,request
# from test import add
from Project_app.utils import IrisDataset

app = Flask(__name__)

@app.route('/') #HomeAPI
def hello_Flask():
    print("Welcome to Flask")
    return "Hello Python"

@app.route('/predict_species')
def predict_species1():
    
    data = request.form
    print('Data is:',data)
    
    SepalLengthCm = eval(data['SepalLengthCm'])
    SepalWidthCm = eval(data['SepalWidthCm'])
    PetalLengthCm = eval(data['PetalLengthCm'])
    PetalWidthCm = eval(data['PetalWidthCm'])
    

    print('')    
    
    iris = IrisDataset(SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm)
    species = iris.predict_species()
    
    return jsonify({'Result':f"Predicted Species: {species}"})

app.run(debug=True)

# app.run(host='',port=config,PORT_NUMBER,debug='False')
# host to give url, port number to give sp. port number 1hr:00