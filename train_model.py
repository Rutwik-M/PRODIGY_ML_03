import os
import cv2
import numpy as np
import joblib
import random
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_DIR = 'dataset/train'
IMG_HEIGHT = 64
IMG_WIDTH = 64
N_SAMPLES = 5000

def load_data():
    print("Loading images and extracting HOG features...")
    X = []
    y = []
    
    files = os.listdir(DATA_DIR)
    files = [f for f in files if f.endswith('.jpg')]
    
    random.shuffle(files)
    
    files = files[:N_SAMPLES]

    for i, file in enumerate(files):
        img_path = os.path.join(DATA_DIR, file)
        img = cv2.imread(img_path)
        
        if img is None: continue
            
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True)
        
        X.append(features)
        
        if 'dog' in file:
            y.append(1)
        else:
            y.append(0)
            
        if i % 500 == 0:
            print(f"Processed {i} images...")

    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = load_data()
    
    print(f"Total samples: {len(X)}")
    print(f"Cats: {np.sum(y==0)}, Dogs: {np.sum(y==1)}")

    if len(np.unique(y)) < 2:
        print("Error: Still only found 1 class. Check dataset path or N_SAMPLES.")
        exit()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training SVM with {len(X_train)} samples...")
    svm = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')
    svm.fit(X_train, y_train)
    
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc * 100:.2f}%")
    
    print("Saving model to svm_model_hog.pkl...")
    joblib.dump(svm, 'svm_model_hog.pkl')
    print("Done!")