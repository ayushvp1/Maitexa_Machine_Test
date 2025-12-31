# Maitexa Machine Test - House Price Prediction

## Project Overview
This project builds a Linear Regression model to predict house prices based on **Area**, **Number of Rooms (Bedrooms)**, and **Location Score**. The dataset includes 3 primary locations (New York, Florida, and Atlanta) ranked by property value to provide a realistic prediction model.

## Model Performance
- **Model Type:** Linear Regression
- **Optimized RMSE:** **~$832,904**
- **Improvement:** Reduced error significantly through outlier removal and realistic geographic data mapping.

## Repository Contents
- `requirements.txt`: Project dependencies for easy setup.
- `predict_final.py`: The main interactive prediction application.
- `predict_prices.py`: Core logic for model training and evaluation.
- `clean_data.py`: Data preprocessing script (Duplicate removal & IQR Outlier handling).
- `house_prices.csv`: Original raw dataset.
- `house_prices_cleaned.csv`: Cleaned dataset used for preliminary training.
- `house_prices_final.csv`: Final processed dataset with realistic location mapping.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the interactive predictor:
   ```bash
   python predict_final.py
   ```

## Status
- [x] Initial Repository Setup
- [x] Data Cleaning & Outlier Removal
- [x] Realistic Geographic Data Mapping
- [x] Linear Regression Model Implementation
- [x] Interactive User Input System
- [x] Final Documentation
