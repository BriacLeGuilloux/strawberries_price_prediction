# Strawberry Price Prediction

This project aims to predict strawberry prices at a 2-week horizon using historical data and weather information. The focus is on demonstrating a structured approach to time series forecasting, from data exploration to model evaluation.


## Data Description

The dataset combines weekly strawberry market prices with meteorological data:
- Temporal range: Multiple years of weekly data
- Features: Weather conditions (temperature, precipitation, etc.)
- Target: Weekly average strawberry prices
- Notable characteristic: Missing summer period data (weeks 24-49)

## Technical Approach

1. **Data Processing**
   - No Handling of missing values, removing them
   - Implemented data scaling

2. **Model Development**
   - Baseline: Simple average-based prediction
   - Time Series: ARIMA modeling
   - Machine Learning: XGBoost with feature engineering

3. **Evaluation**
   - RMSE as primary metric
   - Visual comparison of predictions

## Project Structure

The project is organized into four main notebooks, each focusing on a specific aspect of the analysis:

### 1. Exploratory Data Analysis (1_explore.ipynb)
- Data overview and missing values analysis
- Price distribution
- Seasonal patterns identification
- Key finding: Strong seasonal patterns and missing summer data

### 2. Feature Engineering (2_feature_eng.ipynb)
- Data cleaning and preprocessing
- Feature creation and processing
- Data scaling and export

### 3. Model Implementation (3_model.ipynb)
- Implementation of three different approaches:
  - Naive forecast (baseline)
  - ARIMA model (statistical approach)
  - XGBoost (machine learning approach)
- Model training and prediction generation

### 4. Model Evaluation (4_evaluation.ipynb)
- Comprehensive model comparison
- Performance metrics analysis
- Visualization of predictions


## Project Organization

```
├── data/
│   ├── raw/           # Original dataset
│   ├── processed/     # Cleaned and processed data
│   └── interim/       # Intermediate data
├── models/            # Saved model files
└── src/
    ├── fct_eda.py           # EDA functions
    ├── fct_feature_eng.py   # Feature engineering functions
    ├── fct_model.py         # Model implementation
    ├── fct_evaluation.py    # Evaluation functions
    └── parameter.py         # Project parameters
```

## 🚀 Getting Started (Poetry)

1. Install Poetry (if not already installed)  
👉 https://python-poetry.org/docs/#installation

2. Install dependencies:  
```bash
poetry install
```

3. (Optional but recommended) Launch a shell with the Poetry virtual environment:
```bash
poetry shell
```

4. Run the notebooks in the following order:

* 1_explore.ipynb

* 2_feature_eng.ipynb

* 3_model.ipynb

* 4_evaluation.ipynb