"""
Prediction script for the XGBoost model for strawberry price prediction.
This script handles:
1. Loading the trained XGBoost model
2. Data preprocessing for new data
3. Generating and validating predictions
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, Union
import xgboost as xgb

from fct_feature_eng import preprocessing, scale_df
from parameter import get_dict_params

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_predictions(predictions: np.ndarray, params: Dict) -> bool:
    """
    Validate model predictions against defined constraints.
    
    Args:
        predictions: Array of predictions
        params: Parameter dictionary containing validation rules
        
    Returns:
        bool: True if predictions pass all validation checks
    """
    constraints = params['data_validation']['output_constraints']
    
    # Check for missing values
    if np.isnan(predictions).any():
        logger.error("Predictions contain missing values")
        return False
    
    # Check value ranges
    pred_min = predictions.min()
    pred_max = predictions.max()
    if pred_min < constraints['predictions']['min'] or pred_max > constraints['predictions']['max']:
        logger.error(f"Predictions out of expected range: min={pred_min}, max={pred_max}")
        return False
    
    return True

def load_model(params: Dict) -> xgb.XGBRegressor:
    """
    Load the trained XGBoost model.
    
    Args:
        params: Parameter dictionary containing model paths
        
    Returns:
        xgb.XGBRegressor: Loaded model
    """
    try:
        model = joblib.load(params['paths']['files']['models']['best_model'])
        if not isinstance(model, xgb.XGBRegressor):
            raise ValueError("Loaded model is not an XGBoost model")
        return model
            
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def prepare_data_for_prediction(
    data: pd.DataFrame,
    params: Dict
) -> pd.DataFrame:
    """
    Prepare data for XGBoost prediction.
    
    Args:
        data: Input data
        params: Parameter dictionary
        
    Returns:
        pd.DataFrame: Prepared features ready for prediction
    """
    # Preprocess data
    processed_data = preprocessing(data, is_training=False)
    scaled_data = scale_df(processed_data)
    
    # Create lagged features
    n_lags = params['model_params']['current']['n_lags']
    price_series = scaled_data['price']
    
    # Create lagged features
    df = pd.DataFrame(price_series)
    df.columns = ['y']
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = df['y'].shift(i)
    
    # Drop the target column and handle missing values
    features = df.drop('y', axis=1).fillna(method='bfill')
    return features

def generate_predictions(
    model: xgb.XGBRegressor,
    data: pd.DataFrame
) -> np.ndarray:
    """
    Generate predictions using the XGBoost model.
    
    Args:
        model: XGBoost model
        data: Prepared input data
        
    Returns:
        np.ndarray: Model predictions
    """
    try:
        return model.predict(data)
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise

def main(input_data: Union[str, pd.DataFrame] = None):
    """
    Main prediction pipeline.
    
    Args:
        input_data: Either path to CSV file or pandas DataFrame
    """
    # Load parameters
    params = get_dict_params()
    
    try:
        # Load model
        logger.info("Loading XGBoost model...")
        model = load_model(params)
        
        # Load or use provided data
        if input_data is None:
            logger.info("No input data provided, using default test data...")
            input_data = params['paths']['files']['data']['input']
            
        if isinstance(input_data, str):
            data = pd.read_csv(input_data)
        else:
            data = input_data.copy()
        
        # Prepare data
        logger.info("Preparing data for prediction...")
        prepared_data = prepare_data_for_prediction(data, params)
        
        # Generate predictions
        logger.info("Generating predictions...")
        predictions = generate_predictions(model, prepared_data)
        
        # Validate predictions
        if not validate_predictions(predictions, params):
            raise ValueError("Prediction validation failed")
        
        # Save predictions
        logger.info("Saving predictions...")
        joblib.dump(predictions, params['paths']['files']['models']['predictions'])
        
        logger.info("Prediction completed successfully")
        return predictions
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
