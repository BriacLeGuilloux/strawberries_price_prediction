# fonctions used for notebook 1_explore

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

from src.parameter import get_dict_params

# Load parameters
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


def plot_price_distribution(df: pd.DataFrame) -> None:
    """
    Plot price distribution histogram and time series line plot.
    
    Args:
        df (pd.DataFrame): Input dataframe containing 'price' and 'start_date' columns
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
    Plot correlation heatmap between weather features and price.
    
    Args:
        df (pd.DataFrame): Input dataframe containing weather features and price columns
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
    # Check required columns
    if 'start_date' not in df.columns or 'price' not in df.columns:
        raise ValueError("DataFrame must contain 'start_date' and 'price' columns.")

    # Convert dates and set index
    df = df.copy()
    df['start_date'] = pd.to_datetime(df['start_date'])
    df = df.set_index('start_date')

    # Weekly aggregation and interpolation of missing values
    weekly_price = df['price'].resample('W-MON').mean().interpolate()

    # Additive seasonal decomposition
    decomposition = seasonal_decompose(weekly_price, model='additive', period=52)

    # Plot components
    decomposition.plot()
    plt.suptitle('Seasonal Decomposition of Weekly Strawberry Price', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_seasonal_patterns(df: pd.DataFrame) -> None:
    """
    Plot seasonal patterns in price using monthly and weekly boxplots.
    
    Args:
        df (pd.DataFrame): Input dataframe with 'start_date' and 'price' columns.
                         A 'month' column will be created if it doesn't exist.
    """
    # Add month column if it doesn't exist
    if 'month' not in df.columns:
        df['month'] = df['start_date'].dt.month

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Monthly boxplot
    sns.boxplot(data=df, x='month', y='price', ax=ax1)
    ax1.set_title('Price Distribution by Month')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Price')

    # Weekly boxplot
    sns.boxplot(data=df, x='week', y='price', ax=ax2)
    ax2.set_title('Price Distribution by Week')
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Price')
    ax2.set_xticks(range(1, 54, 2))  # Set readable spacing

    plt.tight_layout()
    plt.show()
