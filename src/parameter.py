# Store parameters in a centralized location. The purpose is to track all variables
# that we might want to modify later and keep them organized in one file.

weather_cols = ['windspeed', 'temp', 'cloudcover', 'precip', 'solarradiation', 'price']
test_start_year = 2022
col_to_scale = ['windspeed', 'temp', 'cloudcover', 'precip', 'solarradiation', 'price']
# ["MAE", "RMSE", "MAPE"]
metric_to_compute = "RMSE"

# Model parameters
model_types = ['naive', 'arima', 'xgboost'] 

def get_dict_params():
    """
    Returns all parameters as a dictionary.
    """
    return {
        'weather_cols': weather_cols,
        'test_start_year': test_start_year,
        'col_to_scale': col_to_scale,
        'model_types': model_types,
        'metric_to_compute': metric_to_compute
    }
