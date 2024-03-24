#Importing necessary libraries

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
#from pydantic import BaseModel
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
import joblib
from pathlib import Path
from task1 import *

app = FastAPI()

#Implementing basic html components for deployment. 
HOME_PAGE =  f"{BASE_DIR}/index.html"

# Read HTML content from file
with open(HOME_PAGE, "r") as file:
    home_page = file.read()


@app.get("/", response_class=HTMLResponse)
async def root():
    return home_page


#Training endpoint for our classifier model. 
@app.post("/api/train")
async def train(dataset_path: str | None = os.path.join(BASE_DIR, 'uWaveGestureLibrary')):
    #Create directory to save model when called train endpoint. 
    os.makedirs('models', exist_ok=True)
    coords, Y = load_data(dataset_path)
    X_train, X_test, y_train, y_test, max_length = prepare_data(coords, Y) 
    model, evaluation_metrics = train_clf(X_train, X_test, y_train, y_test)
    metrics = {'Accuracy' : evaluation_metrics['accuracy'], 
            'Macro Avg' : evaluation_metrics['macro avg'], 
            'Weighted Avg' : evaluation_metrics['weighted avg']}

    return {'metrics' : metrics}


@app.post("/api/predict")
async def inference(file: UploadFile, max_length: int | None = 945):
    #Validation to check if training endpoint has been called before. It is called only if model directory is created.
    try:
        if not os.path.exists(MODEL_DIR):
            raise HTTPException(status_code = 400, detail = "Models not trained yet.")
        input= []
        preds = []
        #Same preprocessing as for the training set without the iteration.
        df = pd.read_csv(file.file, header=None, delimiter=" ")
        reshaped_df = df.values.flatten()
        input = np.array(reshaped_df)
        padding = np.zeros(max_length - len(input))
        input = np.concatenate((input, padding))

        # for loop incase we have several models in the directory. 
        for model in os.listdir(MODEL_DIR):
            clf_model = joblib.load(os.path.join(MODEL_DIR, model))
            pred = clf_model.predict(input.reshape(1, -1))
            preds.append(pred[0])

        return {'Hand Gesture' : str(preds)}
    except Exception as e:
        raise HTTPException(status_code = 404, detail = "File not found.")


