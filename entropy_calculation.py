import numpy as np
import pandas as pd
from scipy.stats import entropy

def shannon_entropy(time_series, bins=10):
    hist, _ = np.histogram(time_series, bins=bins)
    
    prob_dist = hist / hist.sum()
    
    prob_dist = prob_dist[prob_dist > 0]
    
    return entropy(prob_dist)

def sample_entropy(time_series, m=2, r=0.2):
    time_series = np.array(time_series)
    time_series = (time_series - np.mean(time_series)) / np.std(time_series)
    
    if r is None:
        r = 0.2 * np.std(time_series)
        
    N = len(time_series)
    
    A = 0
    B = 0
    
    for i in range(N - m):
        template_m = time_series[i:i+m]
        template_m1 = time_series[i:i+m+1]
        
        for j in range(i+1, N - m):
            dist_m = np.max(np.abs(template_m - time_series[j:j+m]))
            dist_m1 = np.max(np.abs(template_m1 - time_series[j:j+m+1]))
            
            if dist_m <= r:
                B += 1
                if dist_m1 <= r:
                    A += 1
    
    if B == 0:
        return float('inf')
    return -np.log(A / B)

def rolling_entropy(time_series, window_size=20, entropy_func=shannon_entropy):
    rolling_entropy_values = []
    
    for i in range(len(time_series) - window_size + 1):
        window = time_series[i:i+window_size]
        entropy_val = entropy_func(window)
        rolling_entropy_values.append(entropy_val)
    
    return pd.Series(
        rolling_entropy_values, 
        index=pd.Series(time_series).index[window_size-1:]
    )