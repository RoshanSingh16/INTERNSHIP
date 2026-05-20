"""
generate_dataset.py
Generates a realistic synthetic credit card fraud dataset.
Run once before training: python generate_dataset.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_LEGIT   = 9700
N_FRAUD   = 300
N_TOTAL   = N_LEGIT + N_FRAUD

def make_legit(n):
    return pd.DataFrame({
        "amount":        np.random.lognormal(mean=4.0, sigma=1.2, size=n).round(2),
        "hour":          np.random.choice(range(7, 23), size=n),          # daytime
        "day_of_week":   np.random.randint(0, 7, size=n),
        "merchant_cat":  np.random.choice(
                             ["grocery","entertainment","travel","retail","dining"],
                             size=n, p=[0.35,0.20,0.10,0.25,0.10]),
        "distance_km":   np.abs(np.random.normal(15, 10, size=n)).round(1),
        "prev_txn_hrs":  np.abs(np.random.normal(24, 12, size=n)).round(1),
        "v1": np.random.normal(0,   1,  size=n),
        "v2": np.random.normal(0,   1,  size=n),
        "v3": np.random.normal(0.5, 1,  size=n),
        "v4": np.random.normal(0,   0.8,size=n),
        "v5": np.random.normal(0,   1,  size=n),
        "is_fraud": 0,
    })

def make_fraud(n):
    return pd.DataFrame({
        "amount":        np.random.lognormal(mean=5.5, sigma=1.5, size=n).round(2),
        "hour":          np.random.choice(list(range(0, 6)) + list(range(22, 24)), size=n),  # odd hours
        "day_of_week":   np.random.randint(0, 7, size=n),
        "merchant_cat":  np.random.choice(
                             ["grocery","entertainment","travel","retail","dining"],
                             size=n, p=[0.10,0.30,0.35,0.15,0.10]),
        "distance_km":   np.abs(np.random.normal(120, 60, size=n)).round(1),
        "prev_txn_hrs":  np.abs(np.random.normal(1, 2, size=n)).round(1),
        "v1": np.random.normal(-3,  1.5,size=n),
        "v2": np.random.normal(2,   1.5,size=n),
        "v3": np.random.normal(-1,  1.2,size=n),
        "v4": np.random.normal(3,   1,  size=n),
        "v5": np.random.normal(-2,  1.5,size=n),
        "is_fraud": 1,
    })

df = pd.concat([make_legit(N_LEGIT), make_fraud(N_FRAUD)], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("data/creditcard.csv", index=False)
print(f"✅  Dataset saved → data/creditcard.csv  ({len(df)} rows, {df.is_fraud.sum()} fraud)")
