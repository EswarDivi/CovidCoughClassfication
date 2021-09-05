from __future__ import division, print_function
# coding=utf-8
import librosa
import numpy as np
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import load_model
import os
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer



app = Flask(__name__)

model = load_model('Deployment\model.h5')

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    
    
    labelencoder=LabelEncoder()
    
    
    audio, sample_rate = librosa.load(img_path, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    predicted_label=model.predict_classes(mfccs_scaled_features)
    
    if predicted_label==0:
        preds='Positive'
    else:
        preds='Negative stay safe '

    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

      
        preds = model_predict(file_path, model)

        
        result = preds              
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

