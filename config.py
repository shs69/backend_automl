import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")

CLASSIFICATION_MODEL_PATH = os.path.join(MODELS_DIR, "best_classification_model.pkl")
ENCODERS_PATH = os.path.join(MODELS_DIR, "label_encoders.pkl")
REGRESSION_MODEL_PATH = os.path.join(MODELS_DIR, "best_regression_model.pkl")
RESNEXT_MODEL_PATH = os.path.join(MODELS_DIR, "resnext_model.pth")
MASKRCNN_MODEL_PATH = os.path.join(MODELS_DIR, "maskrcnn_model.pth")