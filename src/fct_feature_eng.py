import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew
from src.parameter import get_dict_params
from datetime import datetime

# Load of parameters
dict_params = get_dict_params()
weather_cols = dict_params['weather_cols']
test_start_year = dict_params['test_start_year']
col_to_scale = dict_params['col_to_scale']

def week_to_month(year: int, week: int) -> int:
    # Convertit année + semaine (ISO) en date (lundi de la semaine)
    date = datetime.strptime(f'{year}-W{week:02d}-1', "%Y-W%W-%w")
    return date.month

def create_temporal_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features from date columns
    """
    # time based features
    result = data.assign(
        month=data.apply(lambda row: week_to_month(row['year'], row['week']), axis=1)
    )
    
    # Seasonal features
    result['season'] = pd.cut(result['month'], 
                            bins=[0, 3, 6, 9, 12], 
                            labels=['Winter', 'Spring', 'Summer', 'Fall'])
    
    # Cyclical encoding of temporal features
    result['month_sin'] = np.sin(2 * np.pi * result['month']/12)
    result['month_cos'] = np.cos(2 * np.pi * result['month']/12)
    result['week_sin'] = np.sin(2 * np.pi * result['week']/52)
    result['week_cos'] = np.cos(2 * np.pi * result['week']/52)
    
    return result

def create_weather_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create weather-related features
    """
    # Extreme weather indicators
    # According to internet, temperature above 30°C has negative impact on yield
    result = data.assign(
        is_heatwave=(data['temp'] > 30).astype(int),
        is_windy=(data['windspeed'] > data['windspeed'].quantile(0.75)).astype(int),
        is_rainy=(data['precip'] > 0).astype(int)
    )
    
    return result

def create_lag_features(data: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """
    Create lagged features and rolling statistics
    """
    if cols is None:
        cols = weather_cols
    
    result = data.copy()
    
    # Lagged features
    for col in cols:
        # Previous weeks
        result[f'{col}_lag1'] = result[col].shift(1)
        result[f'{col}_lag2'] = result[col].shift(2)
        result[f'{col}_lag4'] = result[col].shift(4)
        
        # Rolling statistics
        result[f'{col}_rolling_mean_4w'] = result[col].rolling(window=4, min_periods=1).mean()
        result[f'{col}_rolling_std_4w'] = result[col].rolling(window=4, min_periods=1).std()
        result[f'{col}_rolling_max_4w'] = result[col].rolling(window=4, min_periods=1).max()
        result[f'{col}_rolling_min_4w'] = result[col].rolling(window=4, min_periods=1).min()
        
        # Trend indicators
        result[f'{col}_trend'] = result[col] - result[f'{col}_rolling_mean_4w']
        
    return result

def create_price_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create price-specific features
    """
    result = data.copy()
    
    if 'price' in result.columns:
        # Price dynamics
        result['price_momentum'] = result['price'].pct_change()
        result['price_acceleration'] = result['price_momentum'].pct_change()
        result['price_volatility'] = result['price'].rolling(window=4, min_periods=1).std()
        
        # Price relative to historical levels
        result['price_rel_4w_avg'] = result['price'] / result['price'].rolling(window=4, min_periods=1).mean()
        result['price_rel_8w_avg'] = result['price'] / result['price'].rolling(window=8, min_periods=1).mean()
        
        # Seasonal price indicators
        seasonal_avg = result.groupby('month')['price'].transform('mean')
        result['price_rel_seasonal'] = result['price'] / seasonal_avg
        
    return result

def handle_missing_values(data: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Handle missing values in the dataset by dropping rows with missing values
    
    Args:
        data (pd.DataFrame): Input dataframe
        is_training (bool): Whether this is training data
        
    Returns:
        pd.DataFrame: DataFrame with missing values removed
    """
    result = data.copy()
    
    # Drop rows with missing values in weather columns
    result = result.dropna(subset=weather_cols)
    
    return result

def preprocessing(data: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Complete preprocessing pipeline
    
    Args:
        data (pd.DataFrame): Input dataframe
        is_training (bool): Whether this is training data
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    # We remove the rows with week from 24 to 49, as most of them don't contain the prices
    result = data[~data['week'].between(24, 49)]
    
    # Convert dates
    for col in [x for x in result.columns if 'date' in x]:
        result[col] = pd.to_datetime(result[col])
    
    # Handle missing values first
    result = handle_missing_values(result, is_training)
    
    # Create all features
    result = create_temporal_features(result)
    result = create_weather_features(result)
    result = create_lag_features(result)
    result = create_price_features(result)
    
    # Create seasonal indicators
    season_dummies = pd.get_dummies(result['season'], prefix='season')
    result = pd.concat([result, season_dummies], axis=1)
    
    return result

def split_train_test(data: pd.DataFrame, test_start_year: int = test_start_year) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets based on year
    
    Args:
        data (pd.DataFrame): Input dataframe
        test_start_year (int): Year to start test set from
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and testing dataframes
    """
    # Split based on start_date
    train = data[data['year'] < test_start_year].copy()
    test = data[data['year'] >= test_start_year].copy()
    
    return train, test

def scale_df(data: pd.DataFrame) -> pd.DataFrame:
    """
    Scale numerical features in a single DataFrame using StandardScaler.
    Columns to scale are imported from parameter.py
    
    Args:
        data (pd.DataFrame): dataFrame to scale.
        
    Returns:
        pd.DataFrame, df scaled
    """
    result = data.copy()
    col_to_scale = get_dict_params()['col_to_scale']
    
    # Initialisation of scaler
    scaler = StandardScaler()
    result[col_to_scale] = scaler.fit_transform(result[col_to_scale])
    
    return result
