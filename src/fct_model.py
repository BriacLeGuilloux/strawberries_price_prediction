import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
import joblib
from typing import Tuple, Optional
import xgboost as xgb

def naive_forecast(train: pd.Series, test: pd.Series) -> np.ndarray:
    """
    Simple naive forecast using average.
    
    Args:
        train (pd.Series): Training time series
        test (pd.Series): Test time series
        
    Returns:
        np.ndarray: Naive predictions
    """
    last_value = train.mean()
    naive_pred = np.full(len(test), last_value)
    return naive_pred



def fit_arima_model(train: pd.Series, test: pd.Series, order: tuple = (1, 1, 1)) -> Tuple[np.ndarray, Optional[ARIMA]]:
    """
    Fit and predict using ARIMA model
    
    Args:
        train (pd.Series): Training time series
        test (pd.Series): Test time series
        order (tuple): ARIMA order (p,d,q)
        
    Returns:
        tuple: (predictions, fitted model)
    """
    model = ARIMA(train, order=order)
    fitted_model = model.fit()
    predictions = fitted_model.forecast(steps=len(test))
    return predictions, fitted_model

def create_features_for_xgboost(data: pd.Series, n_lags: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create lagged features for XGBoost from time series data
    
    Args:
        data (pd.Series): Time series data
        n_lags (int): Number of lags to use as features
        
    Returns:
        tuple: (X features array, y target array)
    """
    df = pd.DataFrame(data)
    df.columns = ['y']
    
    # Create lagged features
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = df['y'].shift(i)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Split into features and target
    X = df.drop('y', axis=1).values
    y = df['y'].values
    
    return X, y

def fit_xgboost_model(train: pd.Series, test: pd.Series, n_lags: int = 12) -> Tuple[np.ndarray, Optional[xgb.XGBRegressor]]:
    """
    Fit and predict using XGBoost model
    
    Args:
        train (pd.Series): Training time series
        test (pd.Series): Test time series
        n_lags (int): Number of lags to use as features
        
    Returns:
        tuple: (predictions, fitted model)
    """
    # Create features for training
    X_train, y_train = create_features_for_xgboost(train, n_lags)
    
    # Initialize and train XGBoost model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    )
    model.fit(X_train, y_train)
    
    # Create features for prediction
    full_series = pd.concat([train, test])
    X_test, _ = create_features_for_xgboost(full_series, n_lags)
    X_test = X_test[-len(test):]  # Take only the test period
    
    # Make predictions
    predictions = model.predict(X_test)
    
    return predictions, model

def save_model_and_predictions(model: Optional[object], predictions: np.ndarray, model_name: str) -> None:
    """
    Save model and its predictions
    
    Args:
        model: Fitted model
        predictions (np.ndarray): Model predictions
        model_name (str): Name of the model for saving
    """
    if model is not None:
        joblib.dump(model, f'models/{model_name}_model.joblib')
    joblib.dump(predictions, f'models/{model_name}_predictions.joblib')
