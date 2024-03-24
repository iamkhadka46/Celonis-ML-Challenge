#Importing necessary libraries
import os 
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import joblib
from pathlib import Path


BASE_DIR = Path(__file__).resolve(strict=True).parent
MODEL_DIR = os.path.join(BASE_DIR, 'models')


# Loading the dataset from extracted folder. 
def load_data(dataset_path):
    coords = []
    Y = []
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        for file in os.listdir(folder_path):
            if file[0].isalpha() and file.endswith('.txt'):
                # Getting class labels from the file name itself for convinience.
                gesture_index = int(file.split("-")[0][-1])
                df = pd.read_csv(os.path.join(folder_path,file), header=None, delimiter=" ")
                reshaped_df = df.values.flatten()
                coords.append(reshaped_df)
                Y.append(gesture_index)
    return coords, Y


def prepare_data(coords, Y):
    global max_length
    #Getting max length of the sequence for padding. 
    max_length = max(len(seq) for seq in coords)
    #Padding to max length number of dimensions.
    for i in range(len(coords)):
        array_length = len(coords[i])
        padding = np.zeros(max_length - array_length)
        coords[i] = np.concatenate((coords[i], padding))
    #Convert into arrays for model training. 
    X = np.array(coords)
    y = np.array(Y)
    #Split the data in 80/20 percent of train and test set with random state 42 for reproducibility.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, max_length


def train_clf(X_train, X_test, y_train, y_test):
    model = SVC()
    model.fit(X_train, y_train)
    #Getting the path for persisting the model. 
    model_dir = os.path.join(MODEL_DIR, 'clf_model.pkl')
    joblib.dump(model, model_dir)
    y_pred = model.predict(X_test)
    #evaluation_metrics = model.score(X_train, y_train)
    #Creating classification report
    evaluation_metrics = classification_report(y_test, y_pred, output_dict=True)
    return model, evaluation_metrics



# Visualization tool that can be called after production. 
def visualize_evaluation(model, X_test, y_test):
    class_labels = [1, 2, 3, 4, 5, 6, 7, 8]
    y_pred = model.predict(X_test)
    #Get classification report of the model.
    print(classification_report(y_test, y_pred))
    #Creating confusion matrix.
    cm = confusion_matrix(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    cmd.plot()
    plt.show()
    
    
def main():
    dataset_path = ('uWaveGestureLibrary')
    coords, Y = load_data(dataset_path)
    X_train, X_test, y_train, y_test, max_length = prepare_data(coords, Y)
    model, classification_report= train_clf(X_train, X_test, y_train, y_test)
    visualize_evaluation(model, X_test, y_test)


if __name__ == "__main__":
    main()

