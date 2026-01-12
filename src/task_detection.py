def detect_task_type(df, target_column):
    
    unique_values = df[target_column].nunique()
    dtype = df[target_column].dtype

    # Categorical → classification
    if dtype == "object":
        return "classification"

    # Numeric with very few unique values → classification
    if unique_values <= 10:
        return "classification"

    # Otherwise → regression
    return "regression"
