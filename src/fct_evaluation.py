import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from typing import Dict, Any

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from src.parameter import get_dict_params

# Load parameters
dict_params = get_dict_params()
metric_to_compute = dict_params['metric_to_compute']


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray, metric_to_compute="RMSE") -> float:
    """
    Calculate a selected evaluation metric, handling gaps and NaN values in the data.
    
    Args:
        y_true (pd.Series): Actual values
        y_pred (np.ndarray): Predicted values
        dict_params (Dict[str, Any]): Dictionary containing the key 'metric' with value 'MAE', 'RMSE', or 'MAPE'
        
    Returns:
        float: Value of the selected metric
    """
    assert metric_to_compute in ["MAE", "RMSE", "MAPE"], f"Unsupported metric: {metric_to_compute}"

    df = pd.DataFrame({'true': y_true, 'pred': y_pred}, index=y_true.index).dropna()

    if df.empty:
        return np.nan

    gaps = df.index.to_series().diff() > pd.Timedelta(days=31)
    segments = np.cumsum(gaps)

    metric_values = []
    for segment in range(segments.max() + 1):
        segment_data = df[segments == segment]
        if segment_data.empty:
            continue

        try:
            if metric_to_compute == "MAE":
                value = mean_absolute_error(segment_data['true'], segment_data['pred'])
            elif metric_to_compute == "RMSE":
                value = np.sqrt(mean_squared_error(segment_data['true'], segment_data['pred']))
            elif metric_to_compute == "MAPE":
                valid = segment_data['true'] != 0
                if valid.any():
                    value = mean_absolute_percentage_error(
                        segment_data.loc[valid, 'true'],
                        segment_data.loc[valid, 'pred']
                    ) * 100
                else:
                    value = np.nan
            metric_values.append(value)
        except Exception:
            continue

    values = [v for v in metric_values if not np.isnan(v)]
    return np.mean(values) if values else np.nan




def plot_predictions(y_true: pd.Series, predictions: Dict[str, np.ndarray], title: str = "Model Predictions") -> None:
    """
    Plot actual vs predicted values for multiple models, handling gaps in the data
    
    Args:
        y_true (pd.Series): Actual values
        predictions (Dict[str, np.ndarray]): Dictionary of model predictions
        title (str): Plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Plot actual values
    # Convert to DataFrame to easily handle gaps
    actual_df = pd.DataFrame({'value': y_true}, index=y_true.index)
    # Identify gaps (more than 1 month between points)
    gaps = actual_df.index.to_series().diff() > pd.Timedelta(days=31)
    # Split into continuous segments
    segments = np.cumsum(gaps)
    
    for segment in range(segments.max() + 1):
        segment_data = actual_df[segments == segment]
        plt.plot(segment_data.index, segment_data['value'], 
                label='Actual' if segment == 0 else "_nolegend_", 
                linewidth=2)
    
    # Plot predictions
    for name, pred in predictions.items():
        pred_df = pd.DataFrame({'value': pred}, index=y_true.index)
        for segment in range(segments.max() + 1):
            segment_data = pred_df[segments == segment]
            plt.plot(segment_data.index, segment_data['value'], '--',
                    label=name if segment == 0 else "_nolegend_", 
                    alpha=0.7)
    
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_error_distribution(y_true: pd.Series, predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Plot error distributions and return error statistics, handling NaN values
    
    Args:
        y_true (pd.Series): Actual values
        predictions (Dict[str, np.ndarray]): Dictionary of model predictions
        
    Returns:
        Dict[str, Dict[str, float]]: Error statistics by model
    """
    # Calculate errors and remove NaN values
    errors = {}
    for name, pred in predictions.items():
        error = y_true - pred
        errors[name] = error.dropna()
    
    # Plot distributions for models with valid errors
    plt.figure(figsize=(12, 6))
    valid_models = [name for name, error in errors.items() if len(error) > 0]
    
    if valid_models:
        for name in valid_models:
            sns.kdeplot(errors[name], label=name)
        plt.title('Error Distribution by Model')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No valid data for error distribution', 
                ha='center', va='center')
    plt.show()
    
    # Calculate statistics, handling empty or invalid cases
    error_stats = {}
    for name, error in errors.items():
        if len(error) > 0:
            error_stats[name] = {
                'Mean Error': error.mean(),
                'Std Error': error.std(),
                'Max Error': error.abs().max()
            }
        else:
            error_stats[name] = {
                'Mean Error': np.nan,
                'Std Error': np.nan,
                'Max Error': np.nan
            }
    
    return error_stats


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
    # Calculate RMSE for all models
    metrics = {name: calculate_metrics(y_true, pred, metric_to_compute) 
              for name, pred in predictions.items()}
    metrics_df = pd.DataFrame(metrics.items(), columns=['Model', 'RMSE']).set_index('Model')
    
    # Generate all evaluations
    results = {
        'metrics': metrics_df,
        'error_stats': plot_error_distribution(y_true, predictions),
        'rankings': rank_models(metrics_df)
    }
    
    return results
