import pandas as pd
from typing import Dict, Any, Tuple, List
import numpy as np




def test_data_completeness(data: pd.DataFrame, params: Dict) -> Dict:
    """
    Test for missing values and completeness in the dataset.
    
    Args:
        data: Input DataFrame
        params: Parameter dictionary
        
    Returns:
        Dict: Test results
    """
    missing_ratio = data[params['weather_cols']].isnull().mean()
    threshold = params['data_validation']['input_constraints']['missing_threshold']
    
    return {
        'passed': all(missing_ratio <= threshold),
        'details': {col: ratio for col, ratio in missing_ratio.items()}
    }

def test_data_ranges(data: pd.DataFrame, params: Dict) -> Dict:
    """
    Test if values are within expected ranges.
    
    Args:
        data: Input DataFrame
        params: Parameter dictionary
        
    Returns:
        Dict: Test results
    """
    constraints = params['data_validation']['input_constraints']
    range_results = {}
    
    for col in params['weather_cols']:
        if col in constraints:
            values_in_range = (
                (data[col] >= constraints[col]['min']) &
                (data[col] <= constraints[col]['max'])
            ).all()
            range_results[col] = {
                'passed': values_in_range,
                'min': data[col].min(),
                'max': data[col].max()
            }
    
    return range_results

def test_temporal_consistency(data: pd.DataFrame) -> Dict:
    """
    Test temporal consistency of the data.
    
    Args:
        data: Input DataFrame
        
    Returns:
        Dict: Test results
    """
    temporal_consistency = True
    message = ""
    
    for year in data['year'].unique():
        year_data = data[data['year'] == year]
        weeks = year_data['week'].sort_values()
        if not all(weeks.diff().dropna() == 1):
            temporal_consistency = False
            message += f"Non-consecutive weeks found in year {year}\n"
    
    return {
        'passed': temporal_consistency,
        'message': message
    }