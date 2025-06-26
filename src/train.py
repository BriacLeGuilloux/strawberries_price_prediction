"""
Training script for the XGBoost model for strawberry price prediction.
This script handles:
1. Data loading and preprocessing
2. XGBoost model training with hyperparameter optimization
3. Model validation
4. Saving the trained model and its configuration
"""

# Core data & ML
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# Utilities
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

# Optimization & ML
import optuna
import xgboost as xgb

# Personnal functions
from fct_feature_eng import preprocessing, split_train_test, scale_df
from src.fct_train import fit_xgboost_model
from parameter import get_dict_params

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_data(data: pd.DataFrame, params: Dict) -> bool:
    """
    Validate input data against defined constraints.
    
    Args:
        data: Input DataFrame
        params: Parameter dictionary containing validation rules
        
    Returns:
        bool: True if data passes all validation checks
    """
    constraints = params['data_validation']['input_constraints']
    
    # Check missing values
    # TODO
    # missing_ratio = data[params['weather_cols']].isnull().mean().max()
    # if missing_ratio > constraints['missing_threshold']:
    #     logger.error(f"Too many missing values: {missing_ratio:.2%}")
    #     return False
    
    # Validate value ranges
    for col in params['weather_cols']:
        if col in constraints:
            col_min = data[col].min()
            col_max = data[col].max()
            if col_min < constraints[col]['min'] or col_max > constraints[col]['max']:
                logger.error(f"Values out of range for {col}: min={col_min}, max={col_max}")
                return False
    
    return True

def check_model_performance(
    train_rmse: float,
    test_rmse: float
) -> bool:
    """
    Validate model performance against defined thresholds.
    
    Args:
        train_rmse: RMSE on training data
        test_rmse: RMSE on test data
        params: Parameter dictionary containing validation rules
        
    Returns:
        bool: True if model passes all performance checks
    """
    # Load parameters
    params = get_dict_params()
    thresholds = params['model_validation']['performance_thresholds']
    
    # Check for overfitting/underfitting
    train_test_ratio = train_rmse / test_rmse
    # TODO: à reprendre
    # if train_test_ratio > thresholds['max_train_test_rmse_ratio']:
    #     logger.warning(f"Potential overfitting detected: train/test RMSE ratio = {train_test_ratio:.2f}")
    #     return False
    if train_test_ratio < thresholds['min_train_test_rmse_ratio']:
        logger.warning(f"Potential underfitting detected: train/test RMSE ratio = {train_test_ratio:.2f}")
        return False
    
    return True

def optimize_xgboost(trial: optuna.Trial, train_data: pd.Series) -> float:
    """
    Optimize XGBoost hyperparameters using Optuna.
    
    Args:
        trial: Optuna trial object
        train_data: Training data
        
    Returns:
        float: RMSE score
    """
    # Load parameters
    params = get_dict_params()
    ranges = params['model_params']['optimization_ranges']
    
    # Suggest values for XGBoost parameters
    n_estimators = trial.suggest_int('n_estimators', ranges['n_estimators'][0], ranges['n_estimators'][1])
    learning_rate = trial.suggest_float('learning_rate', ranges['learning_rate'][0], ranges['learning_rate'][1])
    max_depth = trial.suggest_int('max_depth', ranges['max_depth'][0], ranges['max_depth'][1])
    n_lags = trial.suggest_int('n_lags', ranges['n_lags'][0], ranges['n_lags'][1])
    min_child_weight = trial.suggest_int('min_child_weight', ranges['min_child_weight'][0], ranges['min_child_weight'][1])
    subsample = trial.suggest_float('subsample', ranges['subsample'][0], ranges['subsample'][1])
    colsample_bytree = trial.suggest_float('colsample_bytree', ranges['colsample_bytree'][0], ranges['colsample_bytree'][1])
    
    # Create validation split
    split_idx = int(len(train_data) * 0.8)
    train_subset = train_data[:split_idx]
    val_subset = train_data[split_idx:]
    
    try:
        predictions, _ = fit_xgboost_model(
            train_subset,
            val_subset,
            n_lags=n_lags,
            params={
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'min_child_weight': min_child_weight,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree
            }
        )
        rmse = np.sqrt(mean_squared_error(val_subset, predictions))
        return rmse
    except:
        return float('inf')

def train_model(
    train_data: pd.Series,
    test_data: pd.Series,
    params: Dict
) -> Tuple[Dict, Dict]:
    """
    Train an XGBoost model with hyperparameter optimization using Optuna.

    Args:
        train_data (pd.Series): Time series training data.
        test_data (pd.Series): Time series testing data.
        params (Dict): Dictionary containing default or fixed model parameters.

    Returns:
        Tuple[Dict, Dict]: Trained model and performance metrics.
    """

    # Step 1: Optimize XGBoost hyperparameters using Optuna
    study = optuna.create_study(direction='minimize')  # We want to minimize the loss (e.g., RMSE)
    study.optimize(lambda trial: optimize_xgboost(trial, train_data), n_trials=20)  # Try 20 combinations
    best_params = study.best_params  # Extract the best parameters found

    # Step 2: Prepare final model parameters
    # Remove 'n_lags' since it's not an XGBoost hyperparameter (used for feature generation)
    model_params = {k: v for k, v in best_params.items() if k != 'n_lags'}
    
    # Merge in default parameters provided externally (e.g., fixed learning rate)
    model_params.update(params['model_params']['current'])

    # Step 3: Train final model and generate predictions
    # 'fit_xgboost_model' handles feature engineering with lags and model training
    predictions, model = fit_xgboost_model(
        train_data,
        test_data,
        n_lags=best_params['n_lags'],
        params=model_params
    )

    # Step 4: Evaluate model performance using RMSE on training and testing sets
    train_pred = model.predict(train_data)  # Predict on training data
    train_rmse = np.sqrt(mean_squared_error(train_data, train_pred))  # Training RMSE
    test_rmse = np.sqrt(mean_squared_error(test_data, predictions))  # Testing RMSE

    # Step 5: Store evaluation metrics and best parameters
    metrics = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'best_params': best_params
    }

    # Return the trained model and evaluation metrics
    return model, metrics


def main():
    """Main training pipeline."""
    # Load parameters
    params = get_dict_params()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_filename = f"xgboost_model_{timestamp}.joblib"
    
    try:
        # Create necessary directories
        logger.info("Creating directories...")
        for category in params['paths']['directories'].values():
            for dir_path in category.values():
                Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        data_path = params['paths']['files']['data']['input']
        raw_data = pd.read_csv(data_path)
        
        # Validate input data
        if not validate_data(raw_data, params):
            raise ValueError("Data validation failed")
        
        # Preprocess data
        processed_data = preprocessing(raw_data)
        train_data, test_data = split_train_test(processed_data, params['test_start_year'])
        
        # Scale data
        train_scaled = scale_df(train_data)
        test_scaled = scale_df(test_data)
        
        # Train model
        logger.info("Training XGBoost model...")
        model, metrics = train_model(
            train_scaled['price'],
            test_scaled['price'],
            params
        )
        
        # Validate model performance
        if not check_model_performance(
            metrics['train_rmse'],
            metrics['test_rmse']
        ):
            raise ValueError("Model validation failed")
        
        # Save model and metrics
        logger.info("Saving model and metrics...")
        joblib.dump(model, params['paths']['files']['models']['best_model'])   # Saving best model
        # joblib.dump(model, model_path)
        
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
