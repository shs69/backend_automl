from tpot import TPOTClassifier
from fastapi import UploadFile, File, HTTPException
from app.utils.preprocessing import preprocess_data, encode_data
from app.utils.load import load_model, load_encoders
import pandas as pd
import logging
import joblib
import config


def train_classification_model(file: UploadFile = File(...), target_column: str = ""):
    try:
        X_train, X_test, y_train, y_test, label_encoders = preprocess_data(file, target_column, task_type="classification")

        model = TPOTClassifier(verbosity=2, random_state=42, generations=5, population_size=20)
        model.fit(X_train, y_train)

        best_model = model.fitted_pipeline_
        joblib.dump(best_model, config.CLASSIFICATION_MODEL_PATH)
        joblib.dump(label_encoders, config.ENCODERS_PATH)

        return {"message": "Модель классификации успешно обучена и сохранена."}
    except Exception as e:
        logging.error(f"Ошибка при обучении модели классификации: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при обучении модели классификации: {str(e)}")
    
def predict_classification_model(file: UploadFile = File(...)):
    model = load_model(config.CLASSIFICATION_MODEL_PATH)
    if model is None:
        raise HTTPException(status_code=404, detail="Модель классификации не обучена. Сначала обучите модель, загрузив набор данных.")

    label_encoders = load_encoders()

    try:
        data = pd.read_csv(file.file)

        data = encode_data(data, label_encoders)

        model_features = model.feature_names_in_
        for feature in model_features:
            if feature not in data.columns:
                raise ValueError(f"Признак '{feature}' отсутствует в данных для предсказания.")

        X = data[model_features]

        predictions = model.predict(X)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        logging.error(f"Ошибка при выполнении предсказаний: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Ошибка при выполнении предсказаний: {str(e)}")