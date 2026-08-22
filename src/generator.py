import numpy as np
import pandas as pd
import os

def generate_biased_data(n_users=5000):
    np.random.seed(42)
    
    # 1. Hidden Confounder: User Loyalty (0 to 100)
    loyalty = np.random.uniform(0, 100, n_users)
    
    # 2. Treatment: Used 'Premium Feature'
    # Logic: Loyal fans are 3x more likely to use it (Selection Bias!)
    treatment_prob = (0.008 * loyalty) + np.random.normal(0, 0.1, n_users)
    treatment_prob = np.clip(treatment_prob, 0, 1)
    used_feature = (np.random.random(n_users) < treatment_prob).astype(int)
    
    # 3. Outcome: Monthly Watch Time (Hours)
    # The TRUTH: The feature only adds exactly 2.0 hours.
    # But loyalty adds 0.5 hours per point.
    watch_time = 10 + (2.0 * used_feature) + (0.5 * loyalty) + np.random.normal(0, 2, n_users)
    
    df = pd.DataFrame({
        'user_id': range(n_users),
        'loyalty_score': loyalty,
        'used_feature': used_feature,
        'watch_time_hrs': watch_time
    })
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/streaming_data.csv', index=False)
    print("✅ Biased dataset generated in /data/streaming_data.csv")

if __name__ == "__main__":
    generate_biased_data()