import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)
model = load_model('model.h5')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Your actual class labels
class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

def predict_image(img_path):
    image = load_img(img_path, target_size=(128, 128))  # ✅ changed from (224, 224) to (128, 128)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) / 255.0        # normalize if model trained on normalized images

    prediction = model.predict(image)[0]
    class_idx = np.argmax(prediction)
    result = class_labels[class_idx]
    confidence = round(float(prediction[class_idx]) * 100, 2)

    return result, confidence

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400
    file = request.files['image']
    if file.filename == '':
        return "No file selected", 400

    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(path)

    result, confidence = predict_image(path)
    return render_template('result.html', result=result, confidence=confidence, image_path=path)

if __name__ == '__main__':
    app.run(debug=True)
