import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_training_data(num_samples=1000):
    # Base locations with more complex patterns
    locations = [
        {'lat': 40.7128, 'lon': -74.0060, 'base_label': 'Low', 'variability': 0.3},      # New York
        {'lat': 34.0522, 'lon': -118.2437, 'base_label': 'Medium', 'variability': 0.4},  # Los Angeles
        {'lat': 51.5074, 'lon': -0.1278, 'base_label': 'High', 'variability': 0.25},     # London
        {'lat': 48.8566, 'lon': 2.3522, 'base_label': 'Medium', 'variability': 0.35},    # Paris
        {'lat': 35.6762, 'lon': 139.6503, 'base_label': 'High', 'variability': 0.3},     # Tokyo
        {'lat': -33.8688, 'lon': 151.2093, 'base_label': 'Low', 'variability': 0.4},     # Sydney
        {'lat': 55.7558, 'lon': 37.6173, 'base_label': 'Medium', 'variability': 0.45},   # Moscow
        {'lat': 1.3521, 'lon': 103.8198, 'base_label': 'High', 'variability': 0.35},     # Singapore
        {'lat': -23.5505, 'lon': -46.6333, 'base_label': 'Low', 'variability': 0.5},     # São Paulo
        {'lat': 25.2048, 'lon': 55.2708, 'base_label': 'Medium', 'variability': 0.3}     # Dubai
    ]
    
    # Generate data
    data = []
    current_year = 2025
    
    for _ in range(num_samples):
        # Randomly select a location
        location = np.random.choice(locations)
        
        # Generate random week (1-52)
        week = np.random.randint(1, 53)
        
        # Add more random variation to coordinates based on location's variability
        lat_variation = np.random.normal(0, location['variability'] * 0.1)
        lon_variation = np.random.normal(0, location['variability'] * 0.1)
        lat = location['lat'] + lat_variation
        lon = location['lon'] + lon_variation
        
        # Add seasonal effects
        seasonal_factor = np.sin(2 * np.pi * week / 52)  # Seasonal pattern
        seasonal_effect = seasonal_factor * 0.2  # Moderate seasonal influence
        
        # Add some randomness to the label based on multiple factors
        base_label = location['base_label']
        label_prob = np.random.random()
        
        # Complex label variation based on multiple factors
        if label_prob < location['variability']:  # Base variability
            if base_label == 'Low':
                if seasonal_effect > 0:  # Positive seasonal effect
                    base_label = np.random.choice(['Low', 'Medium'], p=[0.7, 0.3])
                else:  # Negative seasonal effect
                    base_label = np.random.choice(['Low', 'Medium'], p=[0.9, 0.1])
            elif base_label == 'Medium':
                if seasonal_effect > 0:
                    base_label = np.random.choice(['Low', 'Medium', 'High'], p=[0.2, 0.5, 0.3])
                else:
                    base_label = np.random.choice(['Low', 'Medium', 'High'], p=[0.3, 0.5, 0.2])
            else:  # High
                if seasonal_effect > 0:
                    base_label = np.random.choice(['Medium', 'High'], p=[0.3, 0.7])
                else:
                    base_label = np.random.choice(['Medium', 'High'], p=[0.4, 0.6])
        
        # Add some noise based on distance from base location
        distance_from_base = np.sqrt(lat_variation**2 + lon_variation**2)
        if distance_from_base > 0.2:  # If far from base location
            if np.random.random() < 0.3:  # 30% chance of label change
                base_label = np.random.choice(['Low', 'Medium', 'High'])
        
        data.append({
            'lat': lat,
            'lon': lon,
            'year': current_year,
            'week': week,
            'label': base_label
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('new_training_data.csv', index=False)
    print(f"Generated {num_samples} samples")
    print("\nLabel distribution:")
    print(df['label'].value_counts())
    print("\nSample of data:")
    print(df.head())

if __name__ == "__main__":
    generate_training_data(1000)  # Generate 1000 samples 