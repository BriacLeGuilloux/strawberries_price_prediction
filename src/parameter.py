"""
Centralized configuration for the strawberry price prediction pipeline.
This file contains all parameters and configurations needed for the entire ML pipeline.
Data is collected on a weekly basis.
"""

# Data parameters
weather_cols = ['windspeed', 'temp', 'cloudcover', 'precip', 'solarradiation', 'price']
col_to_scale = ['windspeed', 'temp', 'cloudcover', 'precip', 'solarradiation', 'price']

# Data validation parameters
data_validation = {
    'input_constraints': {
        'missing_threshold': 0.1,  # Maximum allowed proportion of missing values
        'price': {'min': 0, 'max': 1000},  # Expected price range
        'temp': {'min': -10, 'max': 45},   # Expected temperature range
        'windspeed': {'min': 0, 'max': 100},
        'cloudcover': {'min': 0, 'max': 100},
        'precip': {'min': 0, 'max': 500},
        'solarradiation': {'min': 0, 'max': 1500}
    },
    'output_constraints': {
        'predictions': {'min': 0, 'max': 1000},  # Predicted prices should be positive and reasonable
        'allowed_missing': 0.0  # No missing values allowed in predictions
    }
}

# Training parameters
test_start_year = 2022
random_seed = 42
validation_split = 0.2

# Feature engineering parameters for weekly data
lag_features = {
    'short_term': [1, 2, 4],  # Previous weeks (t-1, t-2, t-4 weeks)
    'rolling_window': 4,      # Rolling statistics window (4 weeks)
}

# Models parameters
model_types = ['naive', 'arima', 'xgboost']
# XGBoost model parameters and hyperparameter optimization ranges
model_params = {
    'current': {
        'objective': 'reg:squarederror',
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'n_lags': 12
    },
    'optimization_ranges': {
        'n_estimators': (50, 500),
        'learning_rate': (0.01, 0.3),
        'max_depth': (3, 10),
        'n_lags': (8, 24),
        'min_child_weight': (1, 7),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0)
    }
}

# Model validation parameters
model_validation = {
    'performance_thresholds': {
        'min_rmse_improvement': 0.1,  # Minimum improvement required over baseline
        'max_train_test_rmse_ratio': 1.1,  # Maximum allowed ratio between train and test RMSE (overfitting check)
        'min_train_test_rmse_ratio': 0.9,  # Minimum allowed ratio (underfitting check)
    },
    'cross_validation': {
        'n_splits': 5,
        'gap': 4  # Gap between train and validation sets (in weeks)
    },
    'stability': {
        'max_prediction_shift': 0.2,  # Maximum allowed shift in predictions distribution
        'feature_importance_threshold': 0.05  # Minimum required feature importance
    }
}

# Evaluation parameters
""" 
1. Métriques critiques (bloquantes) :
- Ces métriques doivent absolument être satisfaites pour que le modèle soit mis en production
- Si une de ces métriques échoue, le processus s'arrête avec une erreur
- Exemple : ratio train/test RMSE pour détecter l'overfitting/underfitting

2. Métriques d'avertissement (non-bloquantes) :
- Ces métriques sont importantes mais non critiques
- Si une de ces métriques échoue, un warning est émis mais le processus continue
- Exemple : test de normalité des résidus

"""

metrics = {
    'primary': 'RMSE',  # Root Mean Square Error for model selection
    'critical': {
        'data_validation': True,  # Data must meet all validation criteria
        'prediction_ranges': True,  # Predictions must be within acceptable ranges
        'overfitting_check': True,  # Model must not be overfitting/underfitting
        'min_performance': True,  # Model must meet minimum performance thresholds
    },
    'warning': {
        'residuals_normality': True,  # Check if residuals are normally distributed
        'residuals_autocorrelation': True,  # Check for autocorrelation in residuals
        'feature_importance': True,  # Check feature importance distribution
        'model_stability': True,  # Check model stability across validation folds
    }
}

# Metric thresholds
metric_thresholds = {
    'critical': {
        'min_performance': {
            'max_rmse': 100,  # Maximum acceptable RMSE
            'min_r2': 0.7,    # Minimum acceptable R²
        },
        'prediction_ranges': {
            'min': 0,
            'max': 1000
        },
        'overfitting': {
            'max_train_test_ratio': 1.1,
            'min_train_test_ratio': 0.9
        }
    },
    'warning': {
        'feature_importance': {
            'min_important_features': 3,  # Minimum number of significant features
            'importance_threshold': 0.1   # Minimum importance value for significant features
        },
        'residuals': {
            'normality_pvalue': 0.05,
            'autocorr_threshold': 0.05
        },
        'stability': {
            'max_cv_ratio': 0.2  # Maximum coefficient of variation across folds
        }
    }
}

# Path configurations
paths = {
    'directories': {
        'data': {
            'raw': 'data/raw',
            'processed': 'data/processed',
            'interim': 'data/interim'
        },
        'models': {
            'root': 'models',
            'validation_reports': 'models/validation_reports'
        }
    },
    'files': {
        'data': {
            'input': 'data/raw/senior_ds_test.csv',
            'train_processed': 'data/processed/train_processed.csv',
            'test_processed': 'data/processed/test_processed.csv',
            'train_scaled': 'data/processed/train_scaled.csv',
            'test_scaled': 'data/processed/test_scaled.csv'
        },
        'models': {
            'best_model': 'models/best_model.joblib',
            'predictions': 'models/predictions.joblib',
            'metrics': 'models/metrics.json',
            'evaluation': 'models/validation_reports/evaluation_results.json'
        }
    }
}

def get_dict_params():
    """
    Returns all parameters as a dictionary.
    
    Returns:
        dict: Dictionary containing all configuration parameters
    """
    return {
        # Data parameters
        'weather_cols': weather_cols,
        'col_to_scale': col_to_scale,
        'data_validation': data_validation,
        
        # Training parameters
        'test_start_year': test_start_year,
        'random_seed': random_seed,
        'validation_split': validation_split,
        
        # Feature engineering parameters
        'lag_features': lag_features,
        
        # Model parameters
        'model_types': model_types,
        'model_params': model_params,
        
        # Model validation parameters
        'model_validation': model_validation,
        
        # Evaluation parameters
        'metrics': metrics,
        
        # Paths
        'paths': paths
    }
