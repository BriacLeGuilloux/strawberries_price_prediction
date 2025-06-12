import pandas as pd
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns

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

def split_train_test(df: pd.DataFrame, test_start_year: int = 2022) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

def plot_seasonal_patterns(df: pd.DataFrame) -> None:
    """
    Plot seasonal patterns in price
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Price by month
    price_by_month = df.groupby(df['start_date'].dt.month)['price'].mean()
    price_by_month = price_by_month.reindex(range(1, 13))  # Force les mois 1 à 12, insère NaN si absent
    price_by_month.plot(kind='line', ax=ax1, marker='o')
    ax1.set_title('Average Price by Month')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Average Price')
    
    # Price by week
    df.groupby('week')['price'].mean().plot(
        kind='line', ax=ax2, marker='o'
    )
    ax2.set_title('Average Price by Week')
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Average Price')
    
    plt.tight_layout()
    plt.show()

