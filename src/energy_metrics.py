import time
import psutil
import os
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score
)

# Energy Metrics Module
def measure_performance(model, X_train, y_train, X_test, y_test, task_type):
    process = psutil.Process(os.getpid())

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    memory_usage = process.memory_info().rss / (1024 * 1024)

    start_pred = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_pred

    results = {
        "Training Time (s)": training_time,
        "Inference Time (s)": inference_time,
        "Memory Usage (MB)": memory_usage
    }

    if task_type == "regression":
        results.update({
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
            "R2 Score": r2_score(y_test, y_pred)
        })
    import time
import psutil
import os
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score
)

# Energy Metrics Module
def measure_performance(model, X_train, y_train, X_test, y_test, task_type):
    process = psutil.Process(os.getpid())

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    memory_usage = process.memory_info().rss / (1024 * 1024)

    start_pred = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_pred

    results = {
        "Training Time (s)": training_time,
        "Inference Time (s)": inference_time,
        "Memory Usage (MB)": memory_usage
    }

    if task_type == "regression":
        results.update({
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
            "R2 Score": r2_score(y_test, y_pred)
        })
    else:
        results.update({
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred, average="weighted")
        })

    return results
