import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew
from src.parameter import get_dict_params

# Load of parameters
dict_params = get_dict_params()
weather_cols = dict_params['weather_cols']
test_start_year = dict_params['test_start_year']
col_to_scale = dict_params['col_to_scale']

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

def handle_missing_values(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Handle missing values in the dataset
    """
    df = df.copy()
    
    # For weather features, use rolling mean
    for col in weather_cols:
        if col != 'price':  # Don't impute price for training data
            df[col] = df[col].fillna(df[f'{col}_rolling_mean_4w'])
    
    # For training data, we can drop rows with missing prices
    if is_training:
        df = df.dropna(subset=['price'])
    
    return df

def preprocessing(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Complete preprocessing pipeline
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
    df = handle_missing_values(df, is_training)
    
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
