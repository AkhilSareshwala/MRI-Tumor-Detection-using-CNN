<<<<<<< HEAD
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
model = load_model('model.h5')
class_labels = ['glioma' ,  'meningioma','notumor','pituitary']

def predict_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    if class_labels[predicted_index] == 'notumor':
        result = "No Tumor Detected"
    else:
        result = f"Tumor Type: {class_labels[predicted_index]}"

    return result, confidence * 100

@app.route('/')
def home():
    return render_template('home.html')
=======
import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)
model = load_model('model.h5')
UPLOAD_FOLDER = 'static/uploaded'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')
>>>>>>> da44241 (Docker Image Implementations)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
<<<<<<< HEAD
        return "No file uploaded", 400
    file = request.files['image']
    if file.filename == '':
        return "No file selected", 400
    path = os.path.join('static', 'uploads', file.filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file.save(path)

    result, confidence = predict_image(path)
    return render_template('result.html', result=result, confidence=confidence, image_path=path)
=======
        return "No file part"

    file = request.files['image']
    if file.filename == '':
        return "No selected file"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess image
    img = load_img(filepath, target_size=(128, 128))  # ✅ Correct size for your VGG16-based model
  # adjust if your model needs different size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  

    prediction = model.predict(img_array)[0]
    CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    predicted_class = CLASSES[np.argmax(prediction)]
    confidence = np.max(prediction)

    result = f"Prediction: {predicted_class} (Confidence: {confidence:.2%})"


    return render_template('result.html', prediction=result, image_path=filepath)
>>>>>>> da44241 (Docker Image Implementations)

if __name__ == '__main__':
    app.run(debug=True)
