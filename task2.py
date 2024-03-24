
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
import joblib
from pathlib import Path

app = FastAPI()
BASE_DIR = Path(__file__).resolve(strict=True).parent

MODEL_DIR = os.path.joint(BASE_DIR, 'models')

def load_data(dataset_path):
    coords = []
    Y = []
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        for file in os.listdir(folder_path):
            if file[0].isalpha() and file.endswith('.txt'):
                gesture_index = int(file.split("-")[0][-1])
                df = pd.read_csv(os.path.join(folder_path,file), header=None, delimiter="\s+")
                reshaped_df = df.values.flatten()
                coords.append(reshaped_df)
                Y.append(gesture_index)
    return coords, Y


def prepare_data(coords, Y):
    global max_length
    max_length = max(len(seq) for seq in coords)
    for i in range(len(coords)):
        array_length = len(coords[i])
        padding = np.zeros(max_length - array_length)
        coords[i] = np.concatenate((coords[i], padding))
    X = np.array(coords)
    y = np.array(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, max_length


def train_clf(X_train, X_test, y_train, y_test):
    model = SVC(kernel='rbf', gamma='scale')
    model.fit(X_train, y_train)

    model_dir = os.path.join(MODEL_DIR, 'clf_model.pkl')
    joblib.dump(model, model_dir)

    y_pred = model.predict(X_test)
    #evaluation_metrics = model.score(X_train, y_train)
    evaluation_metrics = classification_report(y_test, y_pred, output_dict=True)
    return model_dir, evaluation_metrics


@app.post("/api/train")
async def train(dataset_path: str | None = os.path.join(BASE_DIR, 'uWaveGestureLibrary')):
    os.makedirs('models', exist_ok=True)
    coords, Y = load_data(dataset_path)
    X_train, X_test, y_train, y_test, max_length = prepare_data(coords, Y) 
    model_dir, evaluation_metrics = train_clf(X_train, X_test, y_train, y_test)
    metrics = {'Accuracy' : evaluation_metrics['accuracy'], 
            'Macro Avg' : evaluation_metrics['macro avg'], 'Weighted Avg' : evaluation_metrics['weighted avg']}

    return {'model_dir' : model_dir, 'metrics' : metrics}


@app.post("/api/predict")
async def inference(file: UploadFile, max_length: int | None = 945):
    if not os.path.exists(MODEL_DIR):
        raise HTTPException(status_code = 400, detail = "Models not trained yet.")
    input= []
    preds = []
    df = pd.read_csv(file.file, header=None, delimiter="\s+")
    reshaped_df = df.values.flatten()
    input = np.array(reshaped_df)
    padding = np.zeros(max_length - len(input))
    input = np.concatenate((input, padding))

    for model in os.listdir(MODEL_DIR):
        clf_model = joblib.load(os.path.join(MODEL_DIR, model))
        pred = clf_model.predict(input.reshape(1, -1))
        preds.append(pred[0])

    return {'Hand Gesture' : str(preds)}


