# Stock Price Prediction App
This project presents a comprehensive approach to stock market prediction by combining machine learning with information-theoretic techniques. The focus is on forecasting the stock prices of Indian companies, with a detailed case study on Reliance Industries Ltd. The core of the project is a Linear Regression model trained on historical price data obtained from the Yahoo Finance API (yfinance). What sets this project apart is the integration of entropy-based features, such as Shannon Entropy and Sample Entropy, which quantify the uncertainty and randomness in stock price movements.

The system supports input through both CSV uploads and live ticker symbols, allowing for flexibility in data analysis. It calculates key technical indicators, performs entropy-based feature engineering, and generates visual comparisons of actual vs. predicted prices using interactive Plotly charts. The backend is built with Flask, while the frontend offers a clean and intuitive UI for uploading data and viewing results.

In addition to the implementation, a detailed research paper has been developed to explain the theoretical underpinnings of the approach, the mathematical formulations of entropy, and their impact on model performance. This project is ideal for anyone interested in quantitative finance, time series forecasting, or applying information theory to real-world datasets.

This is a Streamlit application for predicting stock prices using entropy-based features.

![{F30396BB-22DA-4340-893B-992C945D10F0}](https://github.com/user-attachments/assets/31cd8ef4-53e7-45e5-b78e-e45ab97d7a95)


## Setup Instructions

1. Make sure you have Python 3.11 installed

2. **Set up Virtual Environment (Recommended)**:
   ```bash
   # Open PowerShell as Administrator and navigate to project directory
   cd D:\Capstone\stock_prediction_streamlit

   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   .\venv\Scripts\activate

   # Install requirements
   pip install -r requirements.txt

   # Open VS Code in current directory
   code .
   ```

3. **Running the App**:
   - In VS Code, open a new terminal (Ctrl + `)
   - Make sure you see `(venv)` at the start of your terminal prompt
   - Run the app:
     ```bash
     streamlit run app_streamlit.py
     ```

## Using the stock price prediction

1. Upload a CSV file containing stock data with the following columns:
   - date
   - open
   - high
   - low
   - close
   - volume

2. Adjust the prediction parameters:
   - Prediction Horizon (Days): Number of days to predict into the future
   - Entropy Window Size: Window size for calculating entropy features

3. Click "Run Prediction" to generate predictions

## Troubleshooting

If you encounter any issues:

1. Make sure you're using the virtual environment (you should see `(venv)` in your terminal prompt)
2. Verify all required packages are installed:
   ```bash
   pip list
   ```
3. Check that your CSV file has the correct format
4. Ensure you have enough data points (at least window_size + prediction_horizon)
5. If you get permission errors, try running PowerShell as Administrator


