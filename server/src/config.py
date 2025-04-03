import os

class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, '../models/xgboost_model.json')
    LABEL_ENCODER_PATH = os.path.join(BASE_DIR, '../models/label_encoder.joblib')
    SCALER_PATH = os.path.join(BASE_DIR, '../models/scaler.joblib')
    RETRAIN_DATA_DIR = os.path.join(BASE_DIR, '../retrain_data')
    
    os.makedirs(RETRAIN_DATA_DIR, exist_ok=True)