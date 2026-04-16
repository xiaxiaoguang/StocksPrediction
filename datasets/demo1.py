import sys
sys.path.append('/home/wbyan/stock_data/')

import pandas as pd
import numpy as np
import os
import torch
import pickle
from data_utils import load_data,get_origin_data_from_tushare #会显示报错，但是上面加了sys path可以使用
#读取股市数据
import matplotlib.pyplot as plt
name = "csi500"

def create_stock_data_numpy(save_folder_path, stock_list=None):
    """
    Returns original closing prices in shape (L, N, 1) as a numpy array.
    L: Length, N: Number of Stocks, 1: Single Feature (Price)
    """
    if stock_list is None:
        stock_list = [f.replace('.pkl', '') for f in os.listdir(save_folder_path) if f.endswith('.pkl')]
    
    all_series = []
    valid_stocks = []

    for s in stock_list:
        file_path = os.path.join(save_folder_path, f"{s}.pkl")
        if os.path.exists(file_path):
            df = pd.read_pickle(file_path)
            if 'close' in df.columns:
                # Use the 'close' price
                price_series = df['close']
                all_series.append(price_series)
                valid_stocks.append(s)
    
    # Align stocks by date: Rows = Dates (L), Cols = Stocks (N)
    combined_df = pd.concat(all_series, axis=1, keys=valid_stocks)
    
    # Replace NaN (days with no close price/rest days) with 0
    # This happens directly on the combined_df before numpy conversion
    combined_df = combined_df.fillna(0)

    # Convert to numpy and add the third dimension
    # .values is (L, N), adding np.newaxis makes it (L, N, 1)
    data_bnl = combined_df.values[:, :, np.newaxis]
    
    print(f"Data prepared. Shape: {data_bnl.shape}") 

    return data_bnl, list(combined_df.columns)

# get_origin_data_from_tushare('raw_data/csi500/',stock_list=name,start_date="2005-01-01",end_date="2025-01-01",)
# --- Configuration ---
name = "csi500"
save_path = f'raw_data/{name}/'
tmpl = 96
outl = 3
testl = 245
validl = 245

# Embargo gap to prevent overlap
embargo = tmpl + outl 

# 1. Load the new aligned data
# Assuming features shape is [Time, Assets/Features]
features, stock_list = create_stock_data_numpy(save_path)
# 2. Calculate indices sizes
total_L = features.shape[0]
trainl = int(total_L - testl - validl - (2 * embargo) - tmpl - outl)

# 3. Fit Scaler ONLY on Training Data (Leakage Prevention)
# The training indices will at most reach index: trainl - 1 + tmpl + outl
train_end_idx = trainl + tmpl + outl
train_data = features[:train_end_idx]

# Calculate mean and std (Adding epsilon to prevent division by zero)
# Note: This calculates a global mean/std. If you want per-asset scaling, use axis=0
train_mean = np.mean(train_data)
train_std = np.std(train_data) + 1e-8 

# 4. Apply Standardization
scaled_features = (features - train_mean) / train_std

# 5. Define Labels
# Because your idx tuples are (start, mid, end), your dataloader will grab the future 
# indices automatically. Therefore, labels and features align perfectly at step 0.
scaled_labels = scaled_features.copy()

# 6. Handle folders and saving
if not os.path.exists(f"{name}/"):
    os.makedirs(f"{name}/", exist_ok=True)

with open(f'{name}/data_in{tmpl}_out{outl}.pkl', 'wb') as f:
    pickle.dump({'processed_data': scaled_features}, f)

with open(f'{name}/label_in{tmpl}_out{outl}.pkl', 'wb') as f:
    pickle.dump({'processed_data': scaled_labels}, f)

# 7. Generate Sliding Window Indices
idx = {'train': [], 'valid': [], 'test': []}

# Training Indices
for i in range(0, trainl):
    idx['train'].append((i, i + tmpl, i + tmpl + outl))

# Validation Indices
valid_start = trainl + embargo
for i in range(valid_start, valid_start + validl):
    idx['valid'].append((i, i + tmpl, i + tmpl + outl))

# Test Indices
test_start = valid_start + validl + embargo
for i in range(test_start, test_start + testl):
    idx['test'].append((i, i + tmpl, i + tmpl + outl))

# Save Index
with open(f'{name}/index_in{tmpl}_out{outl}.pkl', 'wb') as f:
    pickle.dump(idx, f)

# 8. Save Actual Scaler Metadata for re_standard_transform
scaler = {
    'func': 're_standard_transform',
    'args': {
        'mean': float(train_mean), 
        'std': float(train_std)
    }
}
with open(f'{name}/scaler_in{tmpl}_out{outl}.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"Processing Complete.")
print(f"Train: {len(idx['train'])}, Valid: {len(idx['valid'])}, Test: {len(idx['test'])}")
print(f"Scaler Mean: {train_mean:.4f}, Scaler Std: {train_std:.4f}")