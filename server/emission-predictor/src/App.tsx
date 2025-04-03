import React, { useState } from 'react';
import { BarChart3, ChevronRight, Upload, BarChart, CheckCircle2 } from 'lucide-react';
import './index.css';

function App() {
  // State for prediction inputs
  const [latitude, setLatitude] = useState('');
  const [longitude, setLongitude] = useState('');
  const [year, setYear] = useState('');
  const [week, setWeek] = useState('');
  const [predictionResult, setPredictionResult] = useState<{
    prediction: string;
    probabilities: number[];
  } | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);

  // State for file upload and training
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingComplete, setTrainingComplete] = useState(false);
  const [trainingMetrics, setTrainingMetrics] = useState<{
    accuracy: number;
    f1_score: number;
    loss: number;
    training_time_seconds: number;
  } | null>(null);
  const [trainingMessage, setTrainingMessage] = useState<string | null>(null);
  const [dataFileName, setDataFileName] = useState<string | null>(null);
  const [trainingTimestamp, setTrainingTimestamp] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  
  // Handle prediction submission
  const handlePredictSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsPredicting(true);
    
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          latitude: parseFloat(latitude),
          longitude: parseFloat(longitude),
          year: parseInt(year),
          week: parseInt(week)
        })
      });
      

      if (!response.ok) {
        throw new Error((await response.json()).error[0]);
      }
      const data = await response.json();
      setPredictionResult({
        prediction: data.prediction,
        probabilities: data.probabilities
      });
    } catch (error) {
      alert(error)
      // You might want to show an error message to the user here
    } finally {
      setIsPredicting(false);
    }
  };
  
  // Handle file upload and training
  const handleFileUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    
    setIsUploading(true);
    setTrainingComplete(false);
    setError(null);
    setUploadProgress(0);
    
    try {
      const formData = new FormData();
      formData.append('file', file);

      const xhr = new XMLHttpRequest();
      
      // Set up progress tracking
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          const progress = (event.loaded / event.total) * 100;
          setUploadProgress(Math.round(progress));
        }
      });

      // Create a promise to handle the XHR request
      const uploadPromise = new Promise((resolve, reject) => {
        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(JSON.parse(xhr.responseText));
          } else {
            reject(new Error(xhr.statusText));
          }
        };
        xhr.onerror = () => reject(new Error('Network error'));
        xhr.ontimeout = () => reject(new Error('Request timed out'));
      });

      // Configure and send the request
      xhr.open('POST', 'http://localhost:5000/retrain');
      xhr.timeout = 300000; // 5 minutes timeout
      xhr.send(formData);

      // Wait for the upload to complete
      const data = await uploadPromise;
      
      setTrainingMessage(data.message);
      setDataFileName(data.data_file);
      setTrainingTimestamp(data.timestamp);
      setTrainingMetrics(data.metrics);
      setTrainingComplete(true);
    } catch (error) {
      console.error('Error:', error);
      setError(error instanceof Error ? error.message : 'An error occurred during upload');
    } finally {
      setIsUploading(false);
      setIsTraining(false);
    }
  };

  // Handle file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-gray-50 to-gray-100 flex items-center justify-center">
      <div className="container w-full max-w-4xl mx-auto py-8 px-4">
        {/* Header */}
        <header className="py-6 mb-8">
          <div className="flex items-center justify-center text-center">
            <div className="flex items-center gap-2">
              <div className="p-2 rounded-lg bg-gradient-primary text-white">
                <BarChart3 size={24} />
              </div>
              <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-cyan-500">
                Emission Predictor
              </h1>
            </div>
          </div>
          <p className="text-center text-gray-500 mt-2 max-w-2xl mx-auto">
            Predict emission values based on location and time data using our advanced machine learning model
          </p>
        </header>
        
        {/* Prediction Section */}
        <div className="mb-8">
          <div className="bg-white rounded-xl p-6 shadow-md">
            <div className="flex items-center mb-4">
              <div className="p-1.5 rounded-md bg-blue-100 text-blue-700 mr-2">
                <BarChart size={18} />
              </div>
              <h2 className="text-xl font-semibold text-gray-800">Make a Prediction</h2>
            </div>
            
            <form onSubmit={handlePredictSubmit} className="space-y-4 text-black">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Latitude
                  </label>
                  <input
                    type="text"
                    value={latitude}
                    onChange={(e) => setLatitude(e.target.value)}
                    placeholder="e.g., 52.520008"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Longitude
                  </label>
                  <input
                    type="text"
                    value={longitude}
                    onChange={(e) => setLongitude(e.target.value)}
                    placeholder="e.g., 13.404954"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Year
                  </label>
                  <input
                    type="number"
                    value={year}
                    onChange={(e) => setYear(e.target.value)}
                    placeholder="e.g., 2023"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Week Number
                  </label>
                  <input
                    type="number"
                    value={week}
                    onChange={(e) => setWeek(e.target.value)}
                    placeholder="e.g., 23"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    required
                    min="1"
                    max="52"
                  />
                </div>
              </div>
              
              <div className="flex justify-end">
                <button
                  type="submit"
                  disabled={isPredicting}
                  className="flex items-center px-4 py-2 bg-gradient-primary text-white rounded-md font-medium hover:opacity-90 transition-opacity disabled:opacity-70"
                >
                  {isPredicting ? (
                    <>
                      <div className="animate-spin mr-2 h-4 w-4 border-2 border-white border-t-transparent rounded-full"></div>
                      Processing...
                    </>
                  ) : (
                    <>
                      Predict
                      <ChevronRight size={18} className="ml-1" />
                    </>
                  )}
                </button>
              </div>
            </form>
            
            {predictionResult !== null && (
              <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-100 animate-fade-in">
                <h3 className="text-lg font-medium text-gray-800 mb-2">Prediction Result</h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Predicted emission level:</span>
                    <span className="text-2xl font-bold text-blue-700">{predictionResult.prediction}</span>
                  </div>
                  <div className="mt-4">
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Probability Distribution:</h4>
                    <div className="grid grid-cols-3 gap-2">
                      {['Low', 'Medium', 'High'].map((level, index) => (
                        <div key={level} className="text-center">
                          <div className="text-sm text-gray-600">{level}</div>
                          <div className="text-lg font-semibold text-blue-700">
                            {(predictionResult.probabilities[index] * 100).toFixed(1)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
                <p className="mt-2 text-xs text-gray-500">
                  This prediction is based on the provided location and time parameters.
                </p>
              </div>
            )}
          </div>
        </div>
        
        {/* Separator */}
        <div className="relative my-8">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-gray-300"></div>
          </div>
        </div>
        
        {/* Model Training Section */}
        <div className="mb-10">
          <div className="bg-white rounded-xl p-6 shadow-md">
            <div className="flex items-center mb-4">
              <div className="p-1.5 rounded-md bg-green-100 text-green-700 mr-2">
                <Upload size={18} />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-gray-800">Model Retraining</h2>
                <p className="text-sm text-gray-500 mt-1">
                  Upload a dataset to retrain the prediction model and improve accuracy
                </p>
              </div>
            </div>
            
            <form onSubmit={handleFileUpload} className="space-y-4">
              <div className="w-full">
                <div className="flex flex-col items-center justify-center p-6 border-2 border-dashed border-gray-300 rounded-lg">
                  <div className="text-center">
                    <Upload className="mx-auto h-12 w-12 text-gray-400 mb-3" />
                    <h3 className="text-sm font-medium text-gray-900">Upload Training Dataset</h3>
                    <p className="text-xs text-gray-500 mt-1">CSV format with labeled data</p>
                    
                    <div className="mt-4">
                      <label className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 cursor-pointer">
                        <span>Select File</span>
                        <input
                          type="file"
                          className="sr-only"
                          accept=".csv"
                          onChange={handleFileChange}
                        />
                      </label>
                    </div>
                    
                    {file && (
                      <p className="mt-2 text-sm text-gray-600">
                        {file.name} ({(file.size / 1024).toFixed(1)} KB)
                      </p>
                    )}
                  </div>
                </div>
              </div>
              
              <div className="flex justify-center">
                <button
                  type="submit"
                  disabled={!file || isUploading || isTraining}
                  className="w-full max-w-xs flex items-center justify-center px-4 py-2 bg-emerald-500 text-white rounded-md font-medium hover:bg-emerald-600 transition-colors disabled:opacity-70"
                >
                  {isUploading ? (
                    <>
                      <div className="animate-spin mr-2 h-4 w-4 border-2 border-white border-t-transparent rounded-full"></div>
                      Uploading... {uploadProgress}%
                    </>
                  ) : isTraining ? (
                    <>
                      <div className="animate-spin mr-2 h-4 w-4 border-2 border-white border-t-transparent rounded-full"></div>
                      Training Model...
                    </>
                  ) : (
                    'Retrain Model'
                  )}
                </button>
              </div>
            </form>
            
            {error && (
              <div className="mt-4 p-4 bg-red-50 rounded-lg border border-red-100">
                <p className="text-red-600 text-sm">{error}</p>
              </div>
            )}
            
            {trainingComplete && trainingMetrics && (
              <div className="mt-6 p-6 bg-green-50 rounded-lg border border-green-100">
                <div className="flex items-center gap-2 mb-4">
                  <CheckCircle2 className="h-5 w-5 text-green-600" />
                  <h3 className="text-lg font-semibold text-gray-800">{trainingMessage}</h3>
                </div>
                
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-gray-600 mb-1">Data File</p>
                      <p className="text-sm font-medium text-gray-900">{dataFileName}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600 mb-1">Timestamp</p>
                      <p className="text-sm font-medium text-gray-900">{trainingTimestamp}</p>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-gray-600 mb-1">Accuracy</p>
                      <p className="text-2xl font-semibold text-gray-900">{(trainingMetrics.accuracy * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600 mb-1">F1 Score</p>
                      <p className="text-2xl font-semibold text-gray-900">{(trainingMetrics.f1_score * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600 mb-1">Loss</p>
                      <p className="text-2xl font-semibold text-gray-900">{trainingMetrics.loss.toFixed(4)}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600 mb-1">Training Time</p>
                      <p className="text-2xl font-semibold text-gray-900">{trainingMetrics.training_time_seconds}s</p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
