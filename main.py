import pandas as pd

from src.data_validation import validate_dataset
from src.preprocessing import preprocess_data
from src.model_training import get_models
from src.energy_metrics import measure_performance
from src.task_detection import detect_task_type


def run_pipeline(file_path, target_column):
    """
    Executes the complete Green AI pipeline:
    - Validation
    - Task detection
    - Preprocessing
    - Model training
    - Energy & accuracy evaluation
    """

    # Validate dataset
    df = validate_dataset(file_path, target_column)

    # Detect task type
    task_type = detect_task_type(df, target_column)

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df, target_column)

    # Load models
    models = get_models(task_type)

    results = []

    for model_name, model in models.items():
        metrics = measure_performance(
            model,
            X_train, y_train,
            X_test, y_test,
            task_type
        )
        metrics["Model"] = model_name
        results.append(metrics)

    return pd.DataFrame(results)
