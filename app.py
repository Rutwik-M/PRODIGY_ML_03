import os
import cv2
import numpy as np
import joblib
from flask import Flask, request, render_template, url_for
from skimage.feature import hog

app = Flask(__name__)

model = joblib.load('svm_model_hog.pkl')

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

IMG_HEIGHT = 64
IMG_WIDTH = 64

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    original_img_url = None
    hog_img_url = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            
            img = cv2.imread(filepath)
            resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            features, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                                      cells_per_block=(2, 2), block_norm='L2-Hys', 
                                      transform_sqrt=True, visualize=True)
            
            features = features.reshape(1, -1)
            pred_prob = model.predict_proba(features)[0]
            
            if pred_prob[1] > pred_prob[0]:
                prediction = "DOG"
                confidence = round(pred_prob[1] * 100, 2)
            else:
                prediction = "CAT"
                confidence = round(pred_prob[0] * 100, 2)

            hog_image_rescaled = (hog_image - hog_image.min()) / (hog_image.max() - hog_image.min()) * 255
            hog_path = os.path.join(UPLOAD_FOLDER, 'hog_' + file.filename)
            cv2.imwrite(hog_path, hog_image_rescaled.astype(np.uint8))
            
            original_img_url = url_for('static', filename=f'uploads/{file.filename}')
            hog_img_url = url_for('static', filename=f'uploads/hog_{file.filename}')

    return render_template('index.html', prediction=prediction, confidence=confidence, 
                           original=original_img_url, hog=hog_img_url)

if __name__ == '__main__':
    app.run(debug=True)