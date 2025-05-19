import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

class EntropyPredictionModel:
    def __init__(self, window_size=20, prediction_horizon=5):
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.model = LinearRegression()
        self.scaler = RobustScaler()
        
    def prepare_features(self, stock_data):
        df = stock_data.copy()
        
        df['returns'] = df['close'].pct_change()
        
        epsilon = 1e-10
        df['returns'] = df['returns'].replace([np.inf, -np.inf], np.nan)
        
        from entropy_calculation import rolling_entropy, shannon_entropy, sample_entropy
        
        try:
            df['shannon_entropy'] = rolling_entropy(
                df['returns'].dropna().clip(-10, 10),
                window_size=self.window_size,
                entropy_func=shannon_entropy
            )
        except Exception as e:
            print(f"Error calculating Shannon entropy: {e}")
            df['shannon_entropy'] = 0
        
        try:
            df['sample_entropy'] = rolling_entropy(
                df['returns'].dropna().clip(-10, 10),
                window_size=self.window_size,
                entropy_func=lambda x: sample_entropy(x, m=2, r=0.2)
            )
        except Exception as e:
            print(f"Error calculating Sample entropy: {e}")
            df['sample_entropy'] = 0
        
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['volume_change'] = df['volume'].pct_change().replace([np.inf, -np.inf], np.nan)
        
        df['target'] = (df['close'].shift(-self.prediction_horizon) / df['close'] - 1).replace([np.inf, -np.inf], np.nan)
        
        df = df.fillna(0)
        
        for col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], 0)
        
        features = ['shannon_entropy', 'sample_entropy', 'sma_20', 'sma_50', 'volume_change', 'returns']
        
        return df[features], df['target']
    
    def train(self, stock_data):
        X, y = self.prepare_features(stock_data)
        
        X = X.replace([np.inf, -np.inf], 0)
        y = y.replace([np.inf, -np.inf], 0)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        try:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        except Exception as e:
            print(f"Error during scaling: {e}")
            X_train_scaled = (X_train - X_train.mean()) / (X_train.std() + 1e-10)
            X_test_scaled = (X_test - X_train.mean()) / (X_train.std() + 1e-10)
            X_train_scaled = np.nan_to_num(X_train_scaled)
            X_test_scaled = np.nan_to_num(X_test_scaled)
        
        try:
            self.model.fit(X_train_scaled, y_train)
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
        except Exception as e:
            print(f"Error during model training: {e}")
            train_score = 0
            test_score = 0
        
        print(f"Training score: {train_score:.4f}")
        print(f"Testing score: {test_score:.4f}")
        
        return train_score, test_score
    
    def predict(self, stock_data):
        X, _ = self.prepare_features(stock_data)
        
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)
        
        try:
            X_scaled = self.scaler.transform(X)
        except Exception as e:
            print(f"Error during prediction scaling: {e}")
            X_scaled = np.nan_to_num((X - X.mean()) / (X.std() + 1e-10))
        
        try:
            predictions = self.model.predict(X_scaled)
            predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.05, neginf=-0.05)
        except Exception as e:
            print(f"Error during prediction: {e}")
            predictions = np.zeros(len(X))
        
        return pd.Series(predictions, index=X.index)