# fonctions used for plotting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any


from src.fct_evaluate_model import split_into_segments



def plot_predictions(y_true: pd.Series, predictions: Dict[str, np.ndarray]) -> None:
    """
    Plot actual vs predicted values for multiple models using consistent colors and segment-aware plotting.
    
    Args:
        y_true (pd.Series): Actual values
        predictions (Dict[str, np.ndarray]): Dictionary of model predictions
    """
    plt.figure(figsize=(12, 6))

    # Palette de couleurs
    color_list = plt.get_cmap("tab10").colors  # 10 couleurs distinctes
    model_names = ['Actual'] + list(predictions.keys())
    color_map = {name: color_list[i % len(color_list)] for i, name in enumerate(model_names)}

    # Tracer les vraies valeurs
    actual_segments = split_into_segments(y_true)
    for i, segment in actual_segments.items():
        plt.plot(segment.index, segment.values,
                 label='Actual' if i == 0 else "_nolegend_",
                 linewidth=2,
                 color=color_map['Actual'])

    # Tracer les prédictions de chaque modèle avec couleur fixe
    for model_name, pred in predictions.items():
        pred_series = pd.Series(pred, index=y_true.index)
        pred_segments = split_into_segments(pred_series)
        for i, segment in pred_segments.items():
            plt.plot(segment.index, segment.values, '--',
                     label=model_name if i == 0 else "_nolegend_",
                     alpha=0.7,
                     color=color_map[model_name])

    plt.title("Model Predictions")
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