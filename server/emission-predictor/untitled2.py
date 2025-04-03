const handlePredict = async () => {
  try {
    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        latitude: 40.7128,
        longitude: -74.0060,
        year: 2025,
        week: 14
      })
    });

    const data = await response.json();
    console.log('Prediction result:', data);
    // Handle the response data here
    
  } catch (error) {
    console.error('Error making prediction:', error);
  }
}; 