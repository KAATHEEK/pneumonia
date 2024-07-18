from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image  # Import Image from PIL (Pillow)

app = Flask(__name__)

# Load your pre-trained model
model_path = 'artifacts/model.h5'
model = load_model(model_path)

# Function to make predictions on a single image
def predict_pneumonia(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image

    prediction = model.predict(img)

    if prediction > 0.5:
        return "Pneumonia (PN)"
    else:
        return "No Pneumonia (Non-PN)"

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            filename = file.filename
            file_path = os.path.join('uploads', filename)
            file.save(file_path)

            result = predict_pneumonia(file_path)
            return render_template('index.html', result=result)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
