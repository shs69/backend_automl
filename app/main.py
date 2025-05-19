import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from app.utils.preprocessing import encode_data
from app.models.classification import train_classification_model, predict_classification_model
from app.models.regression import train_regression_model, predict_regression_model 
from app.models.image_classification import predict_image_classification, load_resnext_model
from app.models.segmentation import load_maskrcnn_model, predict_segmentation
from fastapi.concurrency import run_in_threadpool

@asynccontextmanager
async def lifespan(app: FastAPI):
    global resnext_model, maskrcnn_model
    resnext_model = load_resnext_model()
    maskrcnn_model = load_maskrcnn_model()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logging.basicConfig(level=logging.INFO)

@app.post("/train_classification/")
async def train_classification(file: UploadFile = File(...), target_column: str = ""):
    try:
        return await run_in_threadpool(train_classification_model, file, target_column)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500)

@app.post("/predict_classification/")
async def predict_classification(file: UploadFile = File(...)):
    try:
        return await run_in_threadpool(predict_classification_model, file)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train_regression/")
async def train_regression(file: UploadFile = File(...), target_column: str = ""):
    try:
        return await run_in_threadpool(train_regression_model, file, target_column)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.exception("Unexpected error during regression training")
        raise HTTPException(status_code=500, detail="Unexpected error during regression training")


@app.post("/predict_regression/")
async def predict_regression(file: UploadFile = File(...)):
    try:
        return predict_regression_model(file)
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.exception("Unexpected error during regression training")
        raise HTTPException(status_code=500, detail="Unexpected error during regression training")

    
@app.post("/predict_resnext/")
async def predict_image_classification_route(file: UploadFile = File(...)):
    try:
        resnext_model.eval()
        result = predict_image_classification(file.file, model=resnext_model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict_segmentation/")
async def predict_image_classification_route(file: UploadFile = File(...)):
    try:
        maskrcnn_model.eval()
        result = predict_segmentation(file.file, model=maskrcnn_model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))