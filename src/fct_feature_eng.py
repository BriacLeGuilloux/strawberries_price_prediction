import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew
from src.parameter import get_dict_params

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
    
    # Fallback to global mean for any remaining NaN
    if series_clean.isna().any():
        series_clean = series_clean.fillna(series.mean())
        
    return series_clean

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features from date columns
    """
    df = df.copy()
    
    # Basic time features
    df['year'] = df['start_date'].dt.year
    df['month'] = df['start_date'].dt.month
    df['week'] = df['start_date'].dt.isocalendar().week
    
    # Seasonal features
    df['season'] = pd.cut(df['month'], 
                         bins=[0, 3, 6, 9, 12], 
                         labels=['Winter', 'Spring', 'Summer', 'Fall'])
    
    # Cyclical encoding of temporal features
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['week_sin'] = np.sin(2 * np.pi * df['week']/52)
    df['week_cos'] = np.cos(2 * np.pi * df['week']/52)
    
    return df

def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create weather-related features
    """
    df = df.copy()
    
    # Weather interaction features
    df['temp_precip'] = df['temp'] * df['precip']  # Temperature-precipitation interaction
    df['solar_cloud'] = df['solarradiation'] * (100 - df['cloudcover'])/100  # Effective solar radiation
    df['wind_chill'] = 13.12 + 0.6215 * df['temp'] - 11.37 * (df['windspeed']**0.16) + 0.3965 * df['temp'] * (df['windspeed']**0.16)
    
    # Extreme weather indicators
    df['is_hot'] = (df['temp'] > df['temp'].quantile(0.75)).astype(int)
    df['is_windy'] = (df['windspeed'] > df['windspeed'].quantile(0.75)).astype(int)
    df['is_rainy'] = (df['precip'] > 0).astype(int)
    
    return df

def create_lag_features(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """
    Create lagged features and rolling statistics
    """
    df = df.copy()
    if cols is None:
        cols = weather_cols
    
    # Lagged features
    for col in cols:
        # Previous weeks
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag2'] = df[col].shift(2)
        df[f'{col}_lag4'] = df[col].shift(4)
        
        # Rolling statistics
        df[f'{col}_rolling_mean_4w'] = df[col].rolling(window=4, min_periods=1).mean()
        df[f'{col}_rolling_std_4w'] = df[col].rolling(window=4, min_periods=1).std()
        df[f'{col}_rolling_max_4w'] = df[col].rolling(window=4, min_periods=1).max()
        df[f'{col}_rolling_min_4w'] = df[col].rolling(window=4, min_periods=1).min()
        
        # Trend indicators
        df[f'{col}_trend'] = df[col] - df[f'{col}_rolling_mean_4w']
        
    return df

def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create price-specific features
    """
    df = df.copy()
    
    if 'price' in df.columns:
        # Price dynamics
        df['price_momentum'] = df['price'].pct_change()
        df['price_acceleration'] = df['price_momentum'].pct_change()
        df['price_volatility'] = df['price'].rolling(window=4, min_periods=1).std()
        
        # Price relative to historical levels
        df['price_rel_4w_avg'] = df['price'] / df['price'].rolling(window=4, min_periods=1).mean()
        df['price_rel_8w_avg'] = df['price'] / df['price'].rolling(window=8, min_periods=1).mean()
        
        # Seasonal price indicators
        seasonal_avg = df.groupby('month')['price'].transform('mean')
        df['price_rel_seasonal'] = df['price'] / seasonal_avg
        
    return df

def handle_missing_values(df: pd.DataFrame, is_training: bool = True, 
                         method: str = 'rolling') -> pd.DataFrame:
    """
    Handle missing values in the dataset using specified interpolation method
    
    Args:
        df (pd.DataFrame): Input dataframe
        is_training (bool): Whether this is training data
        method (str): Interpolation method ('mean', 'rolling')
        
    Returns:
        pd.DataFrame: DataFrame with handled missing values
    """
    df = df.copy()
    
    # For weather features, use specified interpolation method
    for col in weather_cols:
        if col != 'price':  # Don't impute price for training data
            df[col] = interpolate_missing_values(df[col], method=method)
    
    # For training data, we can drop rows with missing prices
    if is_training:
        df = df.dropna(subset=['price'])
    
    return df

def preprocessing(df: pd.DataFrame, is_training: bool = True,
                 interpolation_method: str = 'rolling') -> pd.DataFrame:
    """
    Complete preprocessing pipeline
    
    Args:
        df (pd.DataFrame): Input dataframe
        is_training (bool): Whether this is training data
        interpolation_method (str): Method for handling missing values
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    df = df.copy()
    
    # Convert dates
    for col in [x for x in df.columns if 'date' in x]:
        df[col] = pd.to_datetime(df[col])
    
    # Create all features
    df = create_temporal_features(df)
    df = create_weather_features(df)
    df = create_lag_features(df)
    df = create_price_features(df)
    
    # Handle missing values
    df = handle_missing_values(df, is_training, method=interpolation_method)
    
    # Create seasonal indicators
    season_dummies = pd.get_dummies(df['season'], prefix='season')
    df = pd.concat([df, season_dummies], axis=1)
    
    return df

def split_train_test(df: pd.DataFrame, test_start_year: int = test_start_year) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets based on year
    
    Args:
        df (pd.DataFrame): Input dataframe
        test_start_year (int): Year to start test set from
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and testing dataframes
    """
    # Split based on start_date
    train = df[df['year'] < test_start_year].copy()
    test = df[df['year'] >= test_start_year].copy()
    
    return train, test

def scale_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale numerical features in a single DataFrame using StandardScaler.
    Columns to scale are imported from parameter.py
    
    Args:
        df (pd.DataFrame): DataFrame to scale.
        
    Returns:
        pd.DataFrame, df scaled
    """
    df2 = df.copy()
    col_to_scale = get_dict_params()['col_to_scale']
    
    # Initialisation of scaler
    scaler = StandardScaler()
    df2[col_to_scale] = scaler.fit_transform(df2[col_to_scale])
    
    return df2
