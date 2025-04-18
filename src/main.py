import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tpot import TPOTClassifier, TPOTRegressor
from preprocessing import preprocess_data, encode_data

CLASSIFICATION_MODEL_PATH = "best_classification_model.pkl"
REGRESSION_MODEL_PATH = "best_regression_model.pkl"
ENCODERS_PATH = "label_encoders.pkl"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logging.basicConfig(level=logging.INFO)


def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        logging.error("Модель не найдена.")
        return None


def load_encoders():
    try:
        encoders = joblib.load(ENCODERS_PATH)
        return encoders
    except FileNotFoundError:
        logging.error("Энкодеры не найдены.")
        return {}


@app.post("/train_classification/")
async def train_classification(file: UploadFile = File(...), target_column: str = ""):
    try:
        X_train, X_test, y_train, y_test, label_encoders = preprocess_data(file, target_column, task_type="classification")

        model = TPOTClassifier(verbosity=2, random_state=42, generations=5, population_size=20)
        model.fit(X_train, y_train)

        best_model = model.fitted_pipeline_
        joblib.dump(best_model, CLASSIFICATION_MODEL_PATH)
        joblib.dump(label_encoders, ENCODERS_PATH)

        return {"message": "Модель классификации успешно обучена и сохранена."}
    except Exception as e:
        logging.error(f"Ошибка при обучении модели классификации: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при обучении модели классификации: {str(e)}")


@app.post("/predict_classification/")
async def predict_classification(file: UploadFile = File(...)):
    model = load_model(CLASSIFICATION_MODEL_PATH)
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

@app.post("/train_regression/")
async def train_regression(file: UploadFile = File(...), target_column: str = ""):
    try:
        X_train, X_test, y_train, y_test, label_encoders = preprocess_data(file, target_column, task_type="regression")

        model = TPOTRegressor(verbosity=2, random_state=42, generations=10, population_size=50)
        model.fit(X_train, y_train)

        best_model = model.fitted_pipeline_
        joblib.dump(best_model, REGRESSION_MODEL_PATH)
        joblib.dump(label_encoders, ENCODERS_PATH)

        return {"message": "Модель регрессии успешно обучена и сохранена."}
    except Exception as e:
        logging.error(f"Ошибка при обучении модели регрессии: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при обучении модели регрессии: {str(e)}")

@app.post("/predict_regression/")
async def predict_regression(file: UploadFile = File(...)):
    model = load_model(REGRESSION_MODEL_PATH)
    if model is None:
        raise HTTPException(status_code=404, detail="Модель регрессии не обучена. Сначала обучите модель.")

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