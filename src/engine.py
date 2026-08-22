import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def run_causal_analysis():
    df = pd.read_csv('data/streaming_data.csv')
    
    # --- STEP 1: Naive Analysis (The Mistake) ---
    naive_lift = df[df['used_feature']==1]['watch_time_hrs'].mean() - \
                 df[df['used_feature']==0]['watch_time_hrs'].mean()
    
    # --- STEP 2: Propensity Score Matching (The Fix) ---
    scaler = StandardScaler()
    df['loyalty_scaled'] = scaler.fit_transform(df[['loyalty_score']])
    
    treated = df[df['used_feature'] == 1].reset_index(drop=True)
    control = df[df['used_feature'] == 0].reset_index(drop=True)
    
    # Find matching 'Twins' in control group
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(control[['loyalty_scaled']])
    distances, indices = knn.kneighbors(treated[['loyalty_scaled']])
    
    # Get watch time of the 'Twins'
    control_twins_watch_time = control.iloc[indices.flatten()]['watch_time_hrs'].values
    
    # Calculate Average Treatment Effect on the Treated (ATT)
    real_lift = (treated['watch_time_hrs'] - control_twins_watch_time).mean()
    
    print(f"Naive Lift (Wrong): {naive_lift:.2f} hours")
    print(f"Causal Lift (Truth): {real_lift:.2f} hours")
    return naive_lift, real_lift

if __name__ == "__main__":
    run_causal_analysis()