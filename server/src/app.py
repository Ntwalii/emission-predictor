from flask import Flask, request, jsonify
from flask_cors import CORS
import xgboost as xgb
import joblib
import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss
from config import Config
from utils import validate_input, prepare_features

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained model and preprocessing objects
try:
    model = xgb.Booster()
    model.load_model(Config.MODEL_PATH)
    label_encoder = joblib.load(Config.LABEL_ENCODER_PATH)
    scaler = joblib.load(Config.SCALER_PATH)
    logger.info("Model, label encoder, and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or preprocessing objects: {str(e)}")
    raise

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Validate input
        is_valid, error_messages = validate_input(data)
        if not is_valid:
            return jsonify({
                'error': 'Validation failed',
                'details': error_messages
            }), 400
            
        # Prepare features and get feature names
        features, feature_names = prepare_features(data)
        
        # Scale the features
        scaled_features = scaler.transform(features)
        
        # Convert to DMatrix for XGBoost, including feature names
        dmatrix = xgb.DMatrix(scaled_features, feature_names=feature_names)
        
        # Make prediction
        prediction = model.predict(dmatrix)
        
        # For multi-class classification, prediction[0] is an array of probabilities for each class
        probabilities = prediction[0]
        
        # Determine the predicted class using argmax
        predicted_class = int(np.argmax(probabilities))
        
        # Decode the prediction using label encoder
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        
        # Prepare the response with the predicted label and all probabilities
        response = {
            'prediction': predicted_label,
            'probabilities': probabilities.tolist(),
            'input_data': {
                'latitude': data['latitude'],
                'longitude': data['longitude'],
                'year': data['year'],
                'week': data['week']
            }
        }
        
        logger.info(f"Prediction made successfully: {predicted_label}")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': [str(e)]
        }), 500

# Retrain endpoint
@app.route('/retrain', methods=['POST'])
def retrain():
    global model, scaler, label_encoder  # Allow updating the global model and preprocessing objects
    
    try:
        # Log request details
        logger.info(f"Received retrain request with content type: {request.content_type}")
        logger.info(f"Request files: {request.files}")
        logger.info(f"Request headers: {dict(request.headers)}")
        
        if 'file' not in request.files:
            logger.error("No file provided in request")
            return jsonify({
                'error': 'No file provided',
                'details': 'Please upload a CSV file with the field name "file"'
            }), 400
            
        file = request.files['file']
        
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({
                'error': 'No file selected',
                'details': 'Please select a file to upload'
            }), 400
            
        if not file.filename.endswith('.csv'):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({
                'error': 'Invalid file type',
                'details': 'Please upload a CSV file'
            }), 400
        
        # Save uploaded data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"retrain_data_{timestamp}.csv"
        filepath = os.path.join(Config.RETRAIN_DATA_DIR, filename)
        
        try:
            file.save(filepath)
            logger.info(f"File saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            return jsonify({
                'error': 'File upload failed',
                'details': str(e)
            }), 500
        
        # Retraining logic
        def retrain_model(data_path):
            global model, scaler, label_encoder
            
            try:
                # 1. Load new data
                new_data = pd.read_csv(data_path)
                
                # Log the data structure
                logger.info(f"Loaded data shape: {new_data.shape}")
                logger.info(f"Columns in data: {new_data.columns.tolist()}")
                logger.info(f"Sample of data:\n{new_data.head()}")
                
                # Validate required columns
                required_columns = ['lat', 'lon', 'year', 'week', 'label']
                missing_columns = [col for col in required_columns if col not in new_data.columns]
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")
                
                # 2. Preprocess data
                # Separate features and target
                X = new_data[['lat', 'lon', 'year', 'week']]
                y = new_data['label']
                
                # Log class distribution
                logger.info(f"Class distribution:\n{y.value_counts()}")
                
                # Encode the target labels
                y_encoded = label_encoder.transform(y)
                
                # Split into training and validation sets (80-20 split)
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y_encoded, test_size=0.2, random_state=42
                )
                
                # Log split sizes
                logger.info(f"Training set size: {len(X_train)}")
                logger.info(f"Validation set size: {len(X_val)}")
                
                # Scale the features
                X_train_scaled = scaler.fit_transform(X_train)  # Refit scaler on new data
                X_val_scaled = scaler.transform(X_val)
                
                # Convert to DMatrix for XGBoost
                feature_names = ['lat', 'lon', 'year', 'week']
                dtrain = xgb.DMatrix(X_train_scaled, label=y_train, feature_names=feature_names)
                dval = xgb.DMatrix(X_val_scaled, label=y_val, feature_names=feature_names)
                
                # 3. Retrain XGBoost model
                params = {
                    'objective': 'multi:softprob',  # Multi-class classification with probabilities
                    'num_class': len(label_encoder.classes_),
                    'eval_metric': 'mlogloss',  # Multi-class log loss
                    'random_state': 42
                }
                
                # Log model parameters
                logger.info(f"Model parameters: {params}")
                logger.info(f"Number of classes: {len(label_encoder.classes_)}")
                
                # Measure training time
                start_time = datetime.now()
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=100,  # Adjust as needed
                    evals=[(dval, 'validation')],
                    early_stopping_rounds=10,
                    verbose_eval=False
                )
                training_time = (datetime.now() - start_time).total_seconds()
                
                # 4. Evaluate the model on the validation set
                # Get predictions and probabilities
                val_probs = model.predict(dval)
                val_preds = np.argmax(val_probs, axis=1)
                
                # Log prediction distribution
                logger.info(f"Prediction distribution:\n{pd.Series(val_preds).value_counts()}")
                
                # Compute metrics with explicit labels
                labels = np.arange(len(label_encoder.classes_))  # Use all possible classes
                accuracy = accuracy_score(y_val, val_preds)
                f1 = f1_score(y_val, val_preds, average='weighted', labels=labels)
                loss = log_loss(y_val, val_probs, labels=labels)
                
                # Log metrics
                logger.info(f"Validation metrics - Accuracy: {accuracy}, F1: {f1}, Loss: {loss}")
                
                # 5. Save the updated model, scaler, and label encoder
                model.save_model(Config.MODEL_PATH)
                joblib.dump(scaler, Config.SCALER_PATH)
                joblib.dump(label_encoder, Config.LABEL_ENCODER_PATH)
                
                return {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'loss': loss,
                    'training_time_seconds': training_time
                }
                
            except Exception as e:
                logger.error(f"Error during model retraining: {str(e)}")
                raise
        
        # Call the retraining function and get metrics
        metrics = retrain_model(filepath)
        
        # Prepare the response
        response = {
            'message': 'Retraining completed successfully',
            'data_file': filename,
            'timestamp': timestamp,
            'metrics': {
                'accuracy': metrics['accuracy'],
                'f1_score': metrics['f1_score'],
                'loss': metrics['loss'],
                'training_time_seconds': metrics['training_time_seconds']
            }
        }
        
        logger.info(f"Retraining completed with file: {filename}")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        return jsonify({
            'error': 'Retraining failed',
            'details': [str(e)]
        }), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)