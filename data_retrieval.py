import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def get_stock_data(ticker, period='2y'):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        df = df.reset_index()
        df = df.rename(columns={'Date': 'date'})
        
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        
        return df
    
    except Exception as e:
        print(f"Error retrieving data for {ticker}: {e}")
        return None

def save_stock_data(ticker, period='2y', filename=None):
    df = get_stock_data(ticker, period)
    
    if df is None:
        return None
    
    if filename is None:
        filename = f"{ticker}_data.csv"
    
    df.to_csv(filename, index=False)
    
    return filename

if __name__ == "__main__":
    ticker = "AAPL"
    filename = save_stock_data(ticker)
    print(f"Saved data to {filename}")