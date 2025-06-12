weather_cols = ['windspeed', 'temp', 'cloudcover', 'precip', 'solarradiation', 'price']
test_start_year = 2022


def get_dict_params():
    """
    Returns parameters as a dictionary
    """
    return {
        'weather_cols': weather_cols,
        'test_start_year': test_start_year
    }