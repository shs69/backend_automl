import logging
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from fastapi import HTTPException

logging.basicConfig(level=logging.INFO)

def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def preprocess_image_maskrcnn(image_file):
    image = Image.open(image_file).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tensor = transform(image)
    return image_tensor

def preprocess_data(file, target_column, task_type="classification"):
    try:
        data = pd.read_csv(file.file)

        if target_column not in data.columns:
            raise ValueError(f"целевая колонка '{target_column}' не найдена в данных.")

        label_encoders = {}
        for col in data.select_dtypes(include=['object']).columns:
            if col != target_column:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                label_encoders[col] = le

        if task_type == "regression":
            if not pd.api.types.is_numeric_dtype(data[target_column]):
                raise ValueError(f"для задачи регрессии целевая колонка '{target_column}' должна быть числовой.")
        elif task_type == "classification":
            if data[target_column].dtype == "object":
                le = LabelEncoder()
                data[target_column] = le.fit_transform(data[target_column])
                label_encoders[target_column] = le

        X = data.drop(columns=[target_column])
        y = data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test, label_encoders
    except Exception as e:
        logging.error(f"Ошибка при обработке данных: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Ошибка при обработке данных: {str(e)}")


def encode_data(data, label_encoders):
    try:
        for col, encoder in label_encoders.items():
            if col in data.columns:
                data[col] = encoder.transform(data[col])
        return data
    except Exception as e:
        logging.error(f"Ошибка при кодировании данных: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Ошибка при кодировании данных: {str(e)}")