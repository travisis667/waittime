# Taxi Ride Waiting Time Prediction and Recommendation System

## Project Overview
This project predicts taxi ride waiting times using machine learning models (Neural Network and XGBoost) by crawling weather data and preprocessing taxi trip data, helping users optimize their travel decisions. The system includes three core modules: data crawling, preprocessing & visualization, and model training & evaluation.

## Environment Configuration

### Dependency Installation
```bash
pip install -r requirements.txt
```

### Data Preparation
1. Taxi trip data: Store Parquet-format trip data in the `./data` directory
2. Geographical data:
   - Taxi zone information file: `taxi_zone_lookup.csv` (must be in the same directory as scripts)
   - New York taxi zone shapefile (for geographical visualization; path can be adjusted in `solver_recommendation.py`)

## Script Function Description

1. **`load_data.py` - Weather Data Crawling**
   - Function: Automatically crawls hourly weather data (temperature, wind speed, humidity, etc.) for New York from January to June 2024
   - Output: Saved as `new_york_weather_2024_1-6.csv`
   - Execution: `python load_data.py`

2. **`solver_recommendation.py` - Data Preprocessing and Visualization**
   - Function:
     - Loads trip data and weather data, performs cleaning (handles time anomalies, numerical anomalies)
     - Feature engineering (extracts time features, geographical location features, etc.)
     - Generates data visualization charts (feature distribution, correlation heatmaps, waiting time trends, etc.)
   - Output: Preprocessed feature data (saved as Parquet file) and visualization charts
   - Execution: `python solver_recommendation.py`

3. **`model.py` - Model Training and Evaluation**
   - Function:
     - Implements two prediction models: Neural Network (MLP) and XGBoost
     - Supports GPU-accelerated training and automatically selects device (CPU/GPU)
     - Model evaluation and performance comparison
   - Output: Trained model files, evaluation reports, and comparison charts
   - Execution: `python model.py`

## Notes
- When processing large-scale data, adjust the `sample_ratio` parameter in `solver_recommendation.py` to control the sampling proportion
- Model training uses GPU by default (if available), and automatically switches to CPU when no GPU is available
- For first-time execution, run `load_data.py` first to obtain weather data, then run other scripts in sequence