def test_prediction_ranges(predictions: np.ndarray, params: Dict) -> Dict:
    """
    Test if predictions are within expected ranges.
    
    Args:
        predictions: Model predictions
        params: Parameter dictionary
        
    Returns:
        Dict: Test results
    """
    constraints = params['data_validation']['output_constraints']['predictions']
    
    return {
        'passed': (
            (predictions >= constraints['min']) &
            (predictions <= constraints['max'])
        ).all(),
        'min': predictions.min(),
        'max': predictions.max()
    }
    
    
def test_model_stability(
    model: object,
    model_type: str,
    train_data: pd.DataFrame,
    params: Dict
) -> Dict:
    """
    Test model stability using cross-validation.
    
    Args:
        model: Trained model
        model_type: Type of model ('xgboost' or 'arima')
        train_data: Training data
        params: Parameter dictionary
        
    Returns:
        Dict: Test results
    """
    cv_scores = []
    n_splits = params['model_validation']['cross_validation']['n_splits']
    
    # Perform time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in tscv.split(train_data):
        train_subset = train_data.iloc[train_idx]
        val_subset = train_data.iloc[val_idx]
        
        if model_type == 'xgboost':
            predictions = model.predict(val_subset)
        else:  # ARIMA
            predictions = model.forecast(steps=len(val_subset))
            
        rmse = np.sqrt(mean_squared_error(val_subset, predictions))
        cv_scores.append(rmse)
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    return {
        'passed': cv_std / cv_mean < params['model_validation']['stability']['max_prediction_shift'],
        'cv_scores': cv_scores,
        'cv_mean': cv_mean,
        'cv_std': cv_std
    }