import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import config
import os
import urllib
from app.utils.preprocessing import preprocess_image

url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

with urllib.request.urlopen(url) as response:
    CLASS_NAMES = [line.decode('utf-8').strip() for line in response.readlines()]

def load_resnext_model(num_classes=len(CLASS_NAMES)):
    model = models.resnext50_32x4d(pretrained=True)
    torch.save(model.state_dict(), config.RESNEXT_MODEL_PATH)
    return model

def predict_image_classification(image_file, model):
    image_tensor = preprocess_image(image_file)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        class_name = CLASS_NAMES[predicted.item()]
    return {"predictions": [class_name]}
