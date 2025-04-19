import joblib
import logging
import config

def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        logging.error("Модель не найдена.")
        return None

def load_encoders():
    try:
        encoders = joblib.load(config.ENCODERS_PATH)
        return encoders
    except FileNotFoundError:
        logging.error("Энкодеры не найдены.")
        return {}