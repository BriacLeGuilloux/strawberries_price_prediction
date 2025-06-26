"""
Evaluation script for the XGBoost model for strawberry price prediction.
This script handles:
1. Model performance evaluation
2. Unit tests for data and predictions
3. Overfitting/underfitting analysis
4. Model stability checks
5. Feature importance analysis
"""

import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import acorr_ljungbox
import xgboost as xgb

from src.fct_feature_eng import preprocessing, split_train_test, scale_df
from src.parameter import get_dict_params

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_data_completeness(data: pd.DataFrame, params: Dict) -> Dict:
    """
    Test for missing values and completeness in the dataset.
    
    Args:
        data: Input DataFrame
        params: Parameter dictionary
        
    Returns:
        Dict: Test results
    """
    missing_ratio = data[params['weather_cols']].isnull().mean()
    threshold = params['data_validation']['input_constraints']['missing_threshold']
    
    return {
        'passed': all(missing_ratio <= threshold),
        'details': {col: ratio for col, ratio in missing_ratio.items()}
    }

def test_data_ranges(data: pd.DataFrame, params: Dict) -> Dict:
    """
    Test if values are within expected ranges.
    
    Args:
        data: Input DataFrame
        params: Parameter dictionary
        
    Returns:
        Dict: Test results
    """
    constraints = params['data_validation']['input_constraints']
    range_results = {}
    
    for col in params['weather_cols']:
        if col in constraints:
            values_in_range = (
                (data[col] >= constraints[col]['min']) &
                (data[col] <= constraints[col]['max'])
            ).all()
            range_results[col] = {
                'passed': values_in_range,
                'min': data[col].min(),
                'max': data[col].max()
            }
    
    return range_results

def test_temporal_consistency(data: pd.DataFrame) -> Dict:
    """
    Test temporal consistency of the data.
    
    Args:
        data: Input DataFrame
        
    Returns:
        Dict: Test results
    """
    temporal_consistency = True
    message = ""
    
    for year in data['year'].unique():
        year_data = data[data['year'] == year]
        weeks = year_data['week'].sort_values()
        if not all(weeks.diff().dropna() == 1):
            temporal_consistency = False
            message += f"Non-consecutive weeks found in year {year}\n"
    
    return {
        'passed': temporal_consistency,
        'message': message
    }

def test_prediction_ranges(predictions: np.ndarray, params: Dict) -> Dict:
    """
    Test if predictions are within expected ranges.
    
    Args:
        predictions: Model predictions
        params: Parameter dictionary
        
    Returns:
        Dict: Test results
    """
    constraints = params['data_validation']['output_constraints']['predictions']
    
    return {
        'passed': (
            (predictions >= constraints['min']) &
            (predictions <= constraints['max'])
        ).all(),
        'min': predictions.min(),
        'max': predictions.max()
    }

def test_model_stability(
    model: object,
    model_type: str,
    train_data: pd.DataFrame,
    params: Dict
) -> Dict:
    """
    Test model stability using cross-validation.
    
    Args:
        model: Trained model
        model_type: Type of model ('xgboost' or 'arima')
        train_data: Training data
        params: Parameter dictionary
        
    Returns:
        Dict: Test results
    """
    cv_scores = []
    n_splits = params['model_validation']['cross_validation']['n_splits']
    
    # Perform time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in tscv.split(train_data):
        train_subset = train_data.iloc[train_idx]
        val_subset = train_data.iloc[val_idx]
        
        if model_type == 'xgboost':
            predictions = model.predict(val_subset)
        else:  # ARIMA
            predictions = model.forecast(steps=len(val_subset))
            
        rmse = np.sqrt(mean_squared_error(val_subset, predictions))
        cv_scores.append(rmse)
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    return {
        'passed': cv_std / cv_mean < params['model_validation']['stability']['max_prediction_shift'],
        'cv_scores': cv_scores,
        'cv_mean': cv_mean,
        'cv_std': cv_std
    }

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
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(train_data['price'], train_pred))
    test_rmse = np.sqrt(mean_squared_error(test_data['price'], test_pred))
    
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
        }
    }
    
    return evaluation_results

def save_evaluation_results(results: Dict, params: Dict):
    """
    Save evaluation results and generate plots.
    
    Args:
        results: Evaluation results
        params: Parameter dictionary
    """
    # Create validation reports directory if it doesn't exist
    Path(params['paths']['directories']['models']['validation_reports']).mkdir(parents=True, exist_ok=True)
    
    # Save results as JSON
    with open(params['paths']['files']['models']['evaluation'], 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    # Generate and save plots
    plt.figure(figsize=(12, 6))
    sns.histplot(results['model_validation']['residuals']['statistics']['residuals'], kde=True)
    plt.title('Residuals Distribution')
    plt.savefig(f"{params['paths']['directories']['models']['validation_reports']}/residuals_distribution.png")
    plt.close()

def main():
    """Main evaluation pipeline."""
    # Load parameters
    params = get_dict_params()
    
    try:
        # Load data
        logger.info("Loading data...")
        raw_data = pd.read_csv(params['paths']['files']['data']['input'])
        
        # Preprocess data
        processed_data = preprocessing(raw_data)
        train_data, test_data = split_train_test(processed_data, params['test_start_year'])
        
        # Scale data
        train_scaled = scale_df(train_data)
        test_scaled = scale_df(test_data)
        
        # Load model
        logger.info("Loading model...")
        model = joblib.load(params['paths']['files']['models']['best_model'])
        model_type = 'xgboost' if hasattr(model, 'predict') else 'arima'
        
        # Evaluate model
        logger.info("Evaluating model...")
        evaluation_results = evaluate_model_performance(
            model,
            model_type,
            train_scaled,
            test_scaled,
            params
        )
        
        # Save results
        logger.info("Saving evaluation results...")
        save_evaluation_results(evaluation_results, params)
        
        logger.info("Evaluation completed successfully")
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
