import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from typing import Dict, Any

def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate various evaluation metrics
    
    Args:
        y_true (pd.Series): Actual values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    # Preprocess actual values to handle missing data
    
    
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100
    }

def plot_predictions(y_true: pd.Series, predictions: Dict[str, np.ndarray], title: str = "Model Predictions") -> None:
    """
    Plot actual vs predicted values for multiple models
    
    Args:
        y_true (pd.Series): Actual values
        predictions (Dict[str, np.ndarray]): Dictionary of model predictions
        title (str): Plot title
    """
    
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.index, y_true.values, label='Actual', linewidth=2)
    for name, pred in predictions.items():
        plt.plot(y_true.index, pred, '--', label=name, alpha=0.7)
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_error_distribution(y_true: pd.Series, predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Plot error distributions and return error statistics
    
    Args:
        y_true (pd.Series): Actual values
        predictions (Dict[str, np.ndarray]): Dictionary of model predictions
        
    Returns:
        Dict[str, Dict[str, float]]: Error statistics by model
    """
    
    
    # Calculate errors
    errors = {name: y_true - pred for name, pred in predictions.items()}
    
    # Plot distributions
    plt.figure(figsize=(12, 6))
    for name, error in errors.items():
        sns.kdeplot(error, label=name)
    plt.title('Error Distribution by Model')
    plt.legend()
    plt.show()
    
    # Calculate statistics
    error_stats = {name: {
        'Mean Error': error.mean(),
        'Std Error': error.std(),
        'Max Error': error.abs().max()
    } for name, error in errors.items()}
    
    return error_stats

def analyze_seasonal_performance(y_true: pd.Series, predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Analyze and plot seasonal performance
    
    Args:
        y_true (pd.Series): Actual values
        predictions (Dict[str, np.ndarray]): Dictionary of model predictions
        
    Returns:
        pd.DataFrame: Monthly performance metrics
    """
    
    
    # Create seasonal performance DataFrame
    seasonal_performance = pd.DataFrame({
        'Month': y_true.index.month,
        'Actual': y_true.values
    })
    
    for name, pred in predictions.items():
        seasonal_performance[f'{name}_Error'] = np.abs(y_true - pred)
    
    # Plot seasonal errors
    plt.figure(figsize=(12, 6))
    for name in predictions.keys():
        monthly_errors = seasonal_performance.groupby('Month')[f'{name}_Error'].mean()
        plt.plot(monthly_errors.index, monthly_errors.values, 'o-', label=name)
    plt.title('Average Monthly Error by Model')
    plt.xlabel('Month')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.show()
    
    return seasonal_performance.groupby('Month').mean()

def rank_models(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank models based on multiple metrics
    
    Args:
        metrics_df (pd.DataFrame): DataFrame of model metrics
        
    Returns:
        pd.DataFrame: Model rankings
    """
    rankings = metrics_df.rank()
    rankings['Average Rank'] = rankings.mean(axis=1)
    rankings = rankings.sort_values('Average Rank')
    
    # Plot rankings
    plt.figure(figsize=(10, 6))
    rankings['Average Rank'].plot(kind='bar')
    plt.title('Model Rankings (Lower is Better)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return rankings

def evaluate_all_models(y_true: pd.Series, predictions: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Perform comprehensive model evaluation
    
    Args:
        y_true (pd.Series): Actual values
        predictions (Dict[str, np.ndarray]): Dictionary of model predictions
        
    Returns:
        Dict[str, Any]: Complete evaluation results
    """
    # Calculate metrics for all models
    metrics = {name: calculate_metrics(y_true, pred) 
              for name, pred in predictions.items()}
    metrics_df = pd.DataFrame(metrics).T
    
    # Generate all evaluations
    results = {
        'metrics': metrics_df,
        'error_stats': plot_error_distribution(y_true, predictions),
        'seasonal_performance': analyze_seasonal_performance(y_true, predictions),
        'rankings': rank_models(metrics_df)
    }
    
    return results
