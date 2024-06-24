from flask import Flask,request,render_template
import pickle
import numpy as np  

app = Flask(__name__)

def load_model():
    with open('heart.pkl','rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model = data['model']
scaler = data['scaler']

@app.route('/')
def homepage():
    return render_template('heart.html')

@app.route('/predict',methods = ['POST'])
def do_prediction():
    a = request.form.get('age')
    b = request.form.get('anaemia')
    c = request.form.get('creatinine_phosphokinase')
    d = request.form.get('diabetes')
    e = request.form.get('ejection_fraction')
    f = request.form.get('high_blood_pressure')
    g = request.form.get('platelets')
    h = request.form.get('serum_creatinine')
    i = request.form.get('serum_sodium')
    j = request.form.get('sex')
    k = request.form.get('smoking')
    l = request.form.get('time')
    
    b = 1 if b == 'yes' else 0
    d = 1 if d == 'yes' else 0
    f = 1 if f == 'yes' else 0
    j = 1 if j == 'woman' else 0
    k = 1 if k == 'yes' else 0
    
    x = np.array([[a,b,c,d,e,f,g,h,i,j,k,l]])
    x = scaler.transform(x)
    prediction = model.predict(x)
    
    if prediction == 1:
        msg = 'Sorry! patient is likely to pass'
    else:
        msg = 'The patient is well,seek medication ASAP!'
        
    return render_template('heart.html',text = msg,anaemia = b)

if __name__ == '__main__':
    app.run(host="0.0.0.0")
 