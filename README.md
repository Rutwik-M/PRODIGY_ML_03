# 🐶 Cat vs Dog SVM Classifier with HOG Features

**Prodigy InfoTech Internship - Task 03**

This project implements a **Support Vector Machine (SVM)** to classify images of cats and dogs. Unlike standard implementations that flatten raw pixels (which often leads to poor performance), this project utilizes **Histogram of Oriented Gradients (HOG)** for robust feature extraction.

Additionally, the application includes a **Flask-based GUI** that provides "Explainable AI" visualizations, showing users exactly what the computer sees (edges and gradients) alongside the prediction confidence score.

---

## 🚀 Key Features (Innovation)

1.  **HOG Feature Extraction:** Instead of using raw pixel intensity, we extract shape and edge information using HOG, making the model resistant to lighting variations.
2.  **Confidence Scoring:** The model outputs a probability percentage (e.g., "98.5% Dog"), not just a binary class.
3.  **Explainable AI View:** The web interface dynamically generates a visualization of the HOG descriptors, allowing humans to interpret the "computer vision" view of the image.
4.  **Interactive Web UI:** Built with Flask, HTML5, and CSS3.

---

## 📂 Directory Structure

```text
Prodigy_ML_03/
│
├── dataset/                   # (Not included in repo - see Setup)
│   └── train/                 # Contains cat.x.jpg, dog.x.jpg
│
├── static/
│   ├── css/
│   │   └── style.css          # Modern UI styling
│   └── uploads/               # Temp storage for user uploaded images
│
├── templates/
│   └── index.html             # Dashboard with Split-View (Human vs AI)
│
├── app.py                     # Flask Application Logic
├── requirements.txt           # Requirements file
├── train_model.py             # SVM Training Script with HOG
└── README.md                  # Project Documentation
```

## 🛠️ Tech Stack
- Language: Python 3.x
- Web Framework: Flask
- ML Algorithms: SVM (Support Vector Machine) via Scikit-Learn
- Computer Vision: OpenCV, Scikit-Image (HOG)
- Frontend: HTML, CSS

## ⚙️ Installation & Setup
1. Clone the Repository
    ```bash
    git clone [https://github.com/your-username/Prodigy_ML_03.git](https://github.com/your-username/Prodigy_ML_03.git)
    cd Prodigy_ML_03
    ```

2. Install Dependencies
    ```bash
    pip install -r requirements.txt
    ```

3. Setup the Dataset
    1. Download the "Dogs vs Cats" dataset from Kaggle: [Link to Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)
    2. Extract the `train.zip` file.
    3. Create a folder named `dataset` in the project root.
    4. Place the extracted `train` folder inside `dataset/`.
        - Path should look like: `dataset/train/cat.0.jpg`

4. Train the Model
Run the training script to process images, extract HOG features, and save the model as `svm_model_hog.pkl`. (Note: Default is set to 5000 images for quick training. Edit `train_model.py` to increase this).
    ```bash
    python train_model.py
    ```
5. Run the Application
Start the Flask server:
    ```bash
    python app.py
    ```
Open your browser and navigate to: `http://127.0.0.1:5000/`



