import numpy as np
from typing import Dict, Any, Tuple, List
from datetime import datetime

def validate_input(data: Dict[str, Any]) -> tuple[bool, List[str]]:
    errors = []
    required_fields = ['latitude', 'longitude', 'year', 'week']
    
    # Check for missing fields
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        errors.append(f"Missing required fields: {', '.join(missing_fields)}")
        return False, errors
    
    # Validate latitude
    try:
        lat = float(data['latitude'])
        if not (-90 <= lat <= 90):
            errors.append(f"Latitude must be between -90 and 90, got {lat}")
    except ValueError:
        errors.append(f"Invalid latitude format: {data['latitude']}")
    
    # Validate longitude
    try:
        lon = float(data['longitude'])
        if not (-180 <= lon <= 180):
            errors.append(f"Longitude must be between -180 and 180, got {lon}")
    except ValueError:
        errors.append(f"Invalid longitude format: {data['longitude']}")
    
    # Validate year
    try:
        year = int(data['year'])
        current_year = datetime.now().year
        if not (1900 <= year <= current_year):
            errors.append(f"Year must be between 1900 and {current_year}, got {year}")
    except ValueError:
        errors.append(f"Invalid year format: {data['year']}")
    
    # Validate week
    try:
        week = int(data['week'])
        if not (1 <= week <= 52):
            errors.append(f"Week must be between 1 and 52, got {week}")
    except ValueError:
        errors.append(f"Invalid week format: {data['week']}")
    
    return len(errors) == 0, errors

def prepare_features(data: Dict[str, Any]) -> Tuple[np.ndarray, list]:
    features = np.array([
        float(data['latitude']),
        float(data['longitude']),
        int(data['year']),
        int(data['week'])
    ]).reshape(1, -1)
    feature_names = ['lat', 'lon', 'year', 'week']  # Match the names expected by the model
    return features, feature_names