# Stock Price Prediction App

This is a Streamlit application for predicting stock prices using entropy-based features.

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

## Using the App

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