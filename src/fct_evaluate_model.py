# fonctions used for notebook 4_evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from typing import Dict, Any

from parameter import get_dict_params
from fct_plot import plot_error_distribution

# Load parameters
dict_params = get_dict_params()
metric_to_compute = dict_params['metric_to_compute']


def calculate_rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Compute the Root Mean Squared Error (RMSE) between two pandas Series.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """
    Compute the average RMSE over time-consistent segments, handling missing values and temporal gaps.

    Args:
        y_true (pd.Series): Actual values
        y_pred (np.ndarray): Predicted values

    Returns:
        float: Average RMSE across valid segments
    """
    df = pd.DataFrame({'true': y_true, 'pred': y_pred}, index=y_true.index).dropna()

    if df.empty:
        return np.nan

    # Identify temporal gaps longer than 31 days to split the time series
    gaps = df.index.to_series().diff() > pd.Timedelta(days=31)
    segments = np.cumsum(gaps)

    # Calculate RMSE values for each segment
    rmse_values = []
    for segment_id in range(segments.max() + 1):
        segment_data = df[segments == segment_id]
        if not segment_data.empty:
            rmse = calculate_rmse(segment_data['true'], segment_data['pred'])
            rmse_values.append(rmse)

    # Return average RMSE across segments
    return np.mean(rmse_values) if rmse_values else np.nan

def evaluate_all_models(y_true: pd.Series, predictions: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Perform comprehensive model evaluation
    
    Args:
        y_true (pd.Series): Actual values
        predictions (Dict[str, np.ndarray]): Dictionary of model predictions
        
    Returns:
        Dict[str, Any]: Complete evaluation results
    """
    # Calculate RMSE for all models
    metrics = {}
    for name, pred in predictions.items():
        rmse = calculate_metrics(y_true, pred)
        metrics[name] = {'RMSE': rmse}
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Generate all evaluations
    results = {
        'metrics': metrics_df,
        'error_stats': plot_error_distribution(y_true, predictions)
    }
    
    return results


def analyze_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Analyze model residuals for normality and autocorrelation.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dict: Analysis results
    """
    residuals = y_true - y_pred
    
    # Test for normality
    _, normality_pvalue = stats.normaltest(residuals)
    
    # Test for autocorrelation
    acf_test = acorr_ljungbox(residuals, lags=10)
    has_autocorr = any(acf_test.iloc[:, 1] < 0.05)  # p-values < 0.05 indicate autocorrelation
    
    return {
        'normality': {
            'passed': normality_pvalue > 0.05,
            'p_value': normality_pvalue
        },
        'autocorrelation': {
            'passed': not has_autocorr,
            'details': acf_test.to_dict()
        },
        'statistics': {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skew': stats.skew(residuals)
        }
    }
    

def check_overfitting(
    train_rmse: float,
    test_rmse: float,
    params: Dict
) -> Dict:
    """
    Check for overfitting/underfitting using RMSE ratios.
    
    Args:
        train_rmse: RMSE on training data
        test_rmse: RMSE on test data
        params: Parameter dictionary
        
    Returns:
        Dict: Test results
    """
    thresholds = params['model_validation']['performance_thresholds']
    ratio = train_rmse / test_rmse
    
    return {
        'passed': (
            ratio >= thresholds['min_train_test_rmse_ratio'] and
            ratio <= thresholds['max_train_test_rmse_ratio']
        ),
        'ratio': ratio,
        'interpretation': (
            'Overfitting' if ratio > thresholds['max_train_test_rmse_ratio']
            else 'Underfitting' if ratio < thresholds['min_train_test_rmse_ratio']
            else 'Good fit'
        )
    }

def evaluate_model_performance(
    model: object,
    model_type: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    params: Dict
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        model_type: Type of model
        train_data: Training data
        test_data: Test data
        params: Parameter dictionary
        
    Returns:
        Dict: Evaluation results
    """
    # Generate predictions
    if model_type == 'xgboost':
        train_pred = model.predict(train_data)
        test_pred = model.predict(test_data)
    else:  # ARIMA
        train_pred = model.forecast(steps=len(train_data))
        test_pred = model.forecast(steps=len(test_data))
    
    # Calculate metrics using fct_evaluation functions
    train_rmse = calculate_metrics(train_data['price'], train_pred)
    test_rmse = calculate_metrics(test_data['price'], test_pred)
    
    # Plot predictions
    predictions_dict = {'Model': test_pred}
    plot_predictions(test_data['price'], predictions_dict)
    
    # Plot error distribution and get error stats
    error_stats = plot_error_distribution(test_data['price'], predictions_dict)
    
    evaluation_results = {
        'metrics': {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        },
        'data_validation': {
            'completeness': test_data_completeness(test_data, params),
            'ranges': test_data_ranges(test_data, params),
            'temporal': test_temporal_consistency(test_data)
        },
        'prediction_validation': {
            'ranges': test_prediction_ranges(test_pred, params)
        },
        'model_validation': {
            'stability': test_model_stability(model, model_type, train_data, params),
            'residuals': analyze_residuals(test_data['price'], test_pred),
            'overfitting': check_overfitting(train_rmse, test_rmse, params)
        },
        'error_analysis': error_stats
    }
    
    return evaluation_results