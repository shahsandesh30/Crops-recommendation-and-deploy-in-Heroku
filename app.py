import numpy as np
import pickle 
from flask import Flask, request, render_template 

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

model = pickle.load(open('crop_recomd.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def predict():
    features = np.array([float(x) for x in request.form.values()])
    prediction = model.predict(features.reshape(1,7))[0]
        
    return render_template('index.html', prediction_text='Recommended crop is {}'.format(prediction.upper()))

if __name__ == '__main__':
    app.run(port=5000,debug=True)
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=8080)