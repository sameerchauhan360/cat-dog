from tensorflow.keras.models import load_model
import os
from flask import (Flask,
                   render_template,
                   redirect,
                   url_for,
                   request, jsonify)
import numpy as np
from utils.image_processing import preprocess_image

def loadModel(path):
    print('loading')
    model = load_model(path)
    print('done')
    return model

model = loadModel('model/catDogModel.h5')

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
# @app.route('/home')
def home():
    return render_template('index.html')
    
@app.route('/predict', methods=['GET','POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    img_array, img = preprocess_image(file)
    
    prediction = model.predict(img_array)
    predicted_class = (prediction > 0.5).astype("int32")
    class_names = ['Cat', 'Dog']
    result = class_names[predicted_class[0][0]]
    
    img_filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    img.save(img_filename)
    
    return render_template('index.html', prediction=result, 
                           image_url=url_for('static', filename='uploads/' + file.filename))

if __name__=='__main__':
    app.run(debug=True)
