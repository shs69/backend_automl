from tpot import  TPOTRegressor
from fastapi import UploadFile, File, HTTPException
from app.utils.preprocessing import preprocess_data, encode_data
from app.utils.load import load_model, load_encoders
import pandas as pd
import logging
import joblib
import config


def train_regression_model(file: UploadFile = File(...), target_column: str = ""):
    X_train, X_test, y_train, y_test, label_encoders = preprocess_data(file, target_column, task_type="regression")

    model = TPOTRegressor(verbosity=2, random_state=42, generations=10, population_size=50)
    model.fit(X_train, y_train)

    best_model = model.fitted_pipeline_
    joblib.dump(best_model, config.REGRESSION_MODEL_PATH)
    joblib.dump(label_encoders, config.ENCODERS_PATH)

    return {"message": "Модель регрессии успешно обучена и сохранена."}

def predict_regression_model(file: UploadFile = File(...)):
    model = load_model(config.REGRESSION_MODEL_PATH)
    if model is None:
        raise FileNotFoundError("Модель регрессии не обучена. Сначала обучите модель.")

    label_encoders = load_encoders()

    data = pd.read_csv(file.file)
    data = encode_data(data, label_encoders)

    model_features = model.feature_names_in_
    for feature in model_features:
        if feature not in data.columns:
            raise ValueError(f"Признак '{feature}' отсутствует в данных для предсказания.")

    X = data[model_features]
    predictions = model.predict(X)
    return {"predictions": [round(pred, 2) for pred in predictions.tolist()]}
