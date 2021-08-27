from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
import time

from utils.visualization import visualize_classification_2d


def run_model(model, x, y, test_size=0.33, random_state=42):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    start = time.time()
    model.fit(x_train, y_train)
    training_time = time.time() - start

    start = time.time()
    y_pred = model.predict(x_test)
    prediction_time = time.time() - start

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Summary for {model}:")
    print(f"Training time: {training_time}")
    print(f"Prediction time: {prediction_time}")
    print(f"Quality. Accuracy: {accuracy}, Precision: {precision}, Recall {recall}, F1: {f1}")

    visualize_classification_2d(x_train, y_train, model, hint="Train subset")
    visualize_classification_2d(x_test, y_test, model, hint="Test subset")
