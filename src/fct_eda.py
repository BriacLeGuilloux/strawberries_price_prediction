import pandas as pd
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

from src.parameter import get_dict_params

# Load of parameters
dict_params = get_dict_params()
weather_cols = dict_params['weather_cols']
test_start_year = dict_params['test_start_year']


def plot_missing_values(df: pd.DataFrame) -> None:
    """
    Plot missing values heatmap
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull())
    plt.title('Missing Values Heatmap')
    plt.show()

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

def plot_price_distribution(df: pd.DataFrame) -> None:
    """
    Plot price distribution and time series
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    color1 = ['#296C92','#3EB489']
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    
    # Price distribution
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x='price', kde=True)
    plt.title('Distribution: Price')
    
    # Price over time
    plt.subplot(1, 2, 2)
    sns.lineplot(data=df, x='start_date', y='price')
    plt.title('Price vs Date')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_weather_correlations(df: pd.DataFrame) -> None:
    """
    Plot correlation heatmap for weather features and price
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    weather_cols = ['windspeed', 'temp', 'cloudcover', 'precip', 'solarradiation', 'price']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[weather_cols].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Weather Features Correlation with Price')
    plt.show()


def seasonal_analysis_weekly(df: pd.DataFrame) -> None:
    """
    Perform seasonal decomposition of weekly strawberry prices using statsmodels.

    Args:
        df (pd.DataFrame): DataFrame with 'start_date' and 'price' columns (weekly frequency).
    """
    # Vérification des colonnes
    if 'start_date' not in df.columns or 'price' not in df.columns:
        raise ValueError("DataFrame must contain 'start_date' and 'price' columns.")

    # Conversion des dates et indexation
    df = df.copy()
    df['start_date'] = pd.to_datetime(df['start_date'])
    df = df.set_index('start_date')

    # Agrégation par semaine, interpolation des valeurs manquantes
    weekly_price = df['price'].resample('W-MON').mean().interpolate()

    # Décomposition saisonnière additive
    decomposition = seasonal_decompose(weekly_price, model='additive', period=52)

    # Affichage des composantes
    decomposition.plot()
    plt.suptitle('Seasonal Decomposition of Weekly Strawberry Price', fontsize=16)
    plt.tight_layout()
    plt.show()
