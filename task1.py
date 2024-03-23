import os 
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


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


def train_clf(X_train, y_train):
    model = SVC(kernel='rbf', gamma='scale')
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def visualize_evaluation(model, X_test, y_test):
    # Visualize confusion matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(cm))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


def main():
    dataset_path = ('uWaveGestureLibrary')
    coords, Y = load_data(dataset_path)
    X_train, X_test, y_train, y_test, max_length = prepare_data(coords, Y)
    model = train_clf(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    visualize_evaluation(model, X_test, y_test)


if __name__ == "__main__":
    main()

