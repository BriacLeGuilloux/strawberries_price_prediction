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

def interpolate_missing_values(series: pd.Series, 
                             method: str = 'rolling',
                             params: Optional[Dict[str, Any]] = None) -> pd.Series:
    """
    Interpolate missing values using various methods
    
    Args:
        series (pd.Series): Input time series
        method (str): Interpolation method ('mean', 'rolling')
        params (Dict[str, Any], optional): Parameters for the chosen method
            - For 'mean': 
                - groupby: None or str ('year', 'season', 'month')
            - For 'rolling':
                - window: int (default: 4)
                - min_periods: int (default: 1)
                
    Returns:
        pd.Series: Series with interpolated values
    """
    if params is None:
        params = {}
    series_clean = series.copy()
        
    if method == 'mean':
        groupby = params.get('groupby', None)
        if groupby is None:
            # Simple global mean
            fill_value = series.mean()
            series_clean = series.fillna(fill_value)
        else:
            # Grouped mean
            grouped_means = series.groupby(groupby).transform('mean')
            series_clean = series.fillna(grouped_means)
            
    elif method == 'rolling':
        window = params.get('window', 4)
        min_periods = params.get('min_periods', 1)
        rolling_mean = series.rolling(
            window=window,
            min_periods=min_periods,
            center=True
        ).mean()
        series_clean = series.fillna(rolling_mean)
    
    else:
        # no interpolation chosen
        series_clean = series
    
    # Fallback to global mean for any remaining NaN
    if series_clean.isna().any():
        series_clean = series_clean.fillna(series.mean())
        
    return series_clean

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

def handle_missing_values(data: pd.DataFrame, is_training: bool = True, 
                         method: str = 'rolling') -> pd.DataFrame:
    """
    Handle missing values in the dataset using specified interpolation method
    
    Args:
        data (pd.DataFrame): Input dataframe
        is_training (bool): Whether this is training data
        method (str): Interpolation method ('mean', 'rolling')
        
    Returns:
        pd.DataFrame: DataFrame with handled missing values
    """
    result = data.copy()
    
    # For weather features, use specified interpolation method
    for col in weather_cols:
        if col != 'price':  # Don't impute price for training data
            result[col] = interpolate_missing_values(result[col], method=method)
    
    # For training data, we can drop rows with missing prices
    if is_training:
        result = result.dropna(subset=['price'])
    
    return result

def preprocessing(data: pd.DataFrame, is_training: bool = True,
                 interpolation_method: str = 'rolling') -> pd.DataFrame:
    """
    Complete preprocessing pipeline
    
    Args:
        data (pd.DataFrame): Input dataframe
        is_training (bool): Whether this is training data
        interpolation_method (str): Method for handling missing values
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    # We remove the rows with week from 24 to 49, as most of them don't contain the prices
    result = data[~data['week'].between(24, 49)]
    
    # Convert dates
    for col in [x for x in result.columns if 'date' in x]:
        result[col] = pd.to_datetime(result[col])
    
    # Create all features
    result = create_temporal_features(result)
    result = create_weather_features(result)
    result = create_lag_features(result)
    result = create_price_features(result)
    
    # Handle missing values
    result = handle_missing_values(result, is_training, method=interpolation_method)
    
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
