import sys
sys.path.append('/home/wbyan/stock_data/')

import pandas as pd
import numpy as np
import os
import torch
import pickle
#读取股市数据
import matplotlib.pyplot as plt
import itertools
from dataclasses import dataclass
from tqdm import tqdm
import joblib
# ts_pro = ts.pro_api("4883ec2948bef5ea9b8deb81b0c072b1808fb2b1c9d59f0d676a6095")

def create_stock_data_numpy(save_folder_path, stock_list=None, cache_filename="multivariate_cache.pkl"):
    """
    Returns original closing prices in shape (L, N, 1) as a numpy array.
    L: Length, N: Number of Stocks, 1: Single Feature (Price)
    
    Checks for a local cache file first. If missing, aligns individual files 
    and saves the resulting multivariate dataset to disk.
    """
    cache_path = os.path.join(save_folder_path, cache_filename)
    
    # --- 1. CACHE CHECK: Load if exists ---
    if os.path.exists(cache_path):
        print(f"[*] Found existing multivariate cache at {cache_path}. Loading...")
        try:
            # Loading the dictionary we saved earlier
            cached_data = joblib.load(cache_path)
            data_bnl = cached_data['data']
            valid_stocks = cached_data['stocks']
            print(f"[*] Successfully loaded from cache. Shape: {data_bnl.shape} (Timesteps, Stocks, Features)")
            return data_bnl
        except Exception as e:
            print(f"[!] Error loading cache: {e}. Rebuilding from scratch...")
    # --- 2. ORIGINAL LOGIC: Rebuild if no cache found ---
    print("[*] No valid cache found. Aligning individual stocks...")
    
    # Get all available files, making sure NOT to include our cache file by accident
    available_files = [
        f for f in os.listdir(save_folder_path) 
        if f.endswith('.pkl') and f != cache_filename
    ]
    
    file_prefix_map = {f.split('.')[0]: f for f in available_files}

    if stock_list is None:
        matched_files = available_files
    else:
        requested_prefixes = [str(s).split('.')[0] for s in stock_list]
        matched_files = [file_prefix_map[p] for p in requested_prefixes if p in file_prefix_map]

    all_series = []
    valid_stocks = []
    
    for file_name in tqdm(matched_files, desc="Loading stocks"):
        file_path = os.path.join(save_folder_path, file_name)
        stock_name = file_name.replace('.pkl', '')
        
        try:
            df = joblib.load(file_path)
            if 'close' in df.columns:
                # Ensure the index is a DatetimeIndex
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                df.index = df.index.floor('min')
                df = df[~df.index.duplicated(keep='last')]
                df = df.sort_index()
                price_series = df['close']
                all_series.append(price_series)
                valid_stocks.append(stock_name)
        except Exception as e:
            print(f"Error loading {file_name}: {e}")

    if not all_series:
        print("No valid data found.")
        return None
        
    print("Aligning time steps across all stocks...")
    combined_df = pd.concat(all_series, axis=1, keys=valid_stocks)
    
    # Sort the master index to guarantee strict time-step order
    combined_df = combined_df.sort_index()
    
    # Fill missing values
    combined_df = combined_df.ffill().fillna(0)
    
    # Convert to Numpy array with shape (L, N, 1)
    data_bnl = combined_df.values[:318720, :, np.newaxis]
    print(f"Data prepared. Shape: {data_bnl.shape} (Timesteps, Stocks, Features)") 
    
    # --- 3. CACHE SAVE: Store the aligned data ---
    print(f"[*] Saving aligned multivariate data to {cache_path}...")
    try:
        # Saving as a dict so we can keep track of which stocks are at which index
        joblib.dump({'data': data_bnl, 'stocks': valid_stocks}, cache_path)
        print("[*] Save complete!")
    except Exception as e:
        print(f"[!] Warning: Failed to save cache: {e}")

    return data_bnl


def inject_whale_anomalies(data_bnl, num_anomalies=100, z_window=5, 
                                        intensity_range=(0.05, 0.10), stock_ratio_range=(0.1, 0.4)):
    """
    Simulates Whale maneuvers by randomly weighting stocks, calculating strict 
    capital upper bounds to obey the 10% CSI rules, and retrying if the 
    randomized setup cannot generate enough market impact.
    """
    L, N, _ = data_bnl.shape
    new_data = data_bnl.copy()
    max_limit = 0.10  # 10% 涨停 / 跌停
    valid_range = L - (2 * z_window) - 1
    
    successful_injections = 0
    
    for _ in range(num_anomalies):
        success = False
        attempts = 0
        
        while not success:
            attempts += 1
            # Safety breakout: if constraints are too tight, prevent infinite loop
            if attempts > 500:
                # Fallback: relax constraints slightly or skip this anomaly
                break 
                
            # 1. Randomize time and group sizes
            start_t = np.random.randint(0, valid_range)
            stock_ratio = np.random.uniform(*stock_ratio_range)
            num_affected = max(int(N * stock_ratio), 2)
            
            affected_stocks = np.random.choice(N, num_affected, replace=False)
            split_idx = num_affected // 2
            pump_stocks = affected_stocks[:split_idx]
            dump_stocks = affected_stocks[split_idx:]
            
            initial_pump_prices = new_data[start_t, pump_stocks, 0]
            initial_dump_prices = new_data[start_t, dump_stocks, 0]
            
            # 2. Randomize trading ratio/weights (each group sums to 1)
            # Using random uniform and normalizing gives a good spread of weights
            pump_weights = np.random.rand(len(pump_stocks))
            pump_weights /= pump_weights.sum()
            
            dump_weights = np.random.rand(len(dump_stocks))
            dump_weights /= dump_weights.sum()
            
            # 3. Calculate Upper Bounds (Maximum allowed Capital)
            # Formula: Capital * weight_i <= Price_i * 0.10 
            # Therefore: Capital <= (Price_i * 0.10) / weight_i
            max_cap_pump = np.min((initial_pump_prices * max_limit) / pump_weights)
            max_cap_dump = np.min((initial_dump_prices * max_limit) / dump_weights)
            
            # The maximum capital we can move is restricted by the tightest bottleneck
            max_allowed_volume = min(max_cap_pump, max_cap_dump)
            
            # 4. Define Required Intensity (Minimum required Capital)
            # Assuming intensity means the total capital relative to the pump group's total value
            total_pump_value = np.sum(initial_pump_prices)
            min_req_volume = intensity_range[0] * total_pump_value
            max_req_volume = intensity_range[1] * total_pump_value
            
            # 5. The Core Check: Can this randomized setup achieve the required intensity?
            if max_allowed_volume < min_req_volume:
                # The bottleneck hit the 10% limit before we could invest enough money.
                # Retry the randomization!
                continue
                
            # 6. Success! We have a valid trading window. 
            # Randomize final trading volume within the valid intersection of bounds.
            upper_bound_volume = min(max_req_volume, max_allowed_volume)
            final_volume = np.random.uniform(min_req_volume, upper_bound_volume)
            
            # 7. Apply Phase 1 (Perfectly Zero-Sum)
            # Dollar amount added to each stock = final_volume * weight
            new_data[start_t + z_window, pump_stocks, 0] = initial_pump_prices + (final_volume * pump_weights)
            new_data[start_t + z_window, dump_stocks, 0] = initial_dump_prices - (final_volume * dump_weights)
            
            # 8. Apply Phase 2 Follow-through (Optional, using new baseline prices)
            follow_through_noise = np.random.uniform(0.4, 0.8) # Weaker follow-through
            target_p2_volume = final_volume * follow_through_noise
            
            phase1_pump_prices = new_data[start_t + z_window, pump_stocks, 0]
            phase1_dump_prices = new_data[start_t + z_window, dump_stocks, 0]
            
            # Recalculate bottlenecks for Day 2
            max_cap_pump_p2 = np.min((phase1_pump_prices * max_limit) / pump_weights)
            max_cap_dump_p2 = np.min((phase1_dump_prices * max_limit) / dump_weights)
            max_allowed_volume_p2 = min(max_cap_pump_p2, max_cap_dump_p2)
            
            final_volume_p2 = min(target_p2_volume, max_allowed_volume_p2)
            
            new_data[start_t + 2*z_window, pump_stocks, 0] = phase1_pump_prices + (final_volume_p2 * pump_weights)
            new_data[start_t + 2*z_window, dump_stocks, 0] = phase1_dump_prices - (final_volume_p2 * dump_weights)
            
            success = True
            successful_injections += 1
            
    print(f"--- Successfully Injected {successful_injections}/{num_anomalies} Weighted Whale Anomalies ---")
    return new_data

def generate_enhanced_anomaly_datasets(features, save_path, seq_len=12, testl=500, validl=500, whale_params=None, balance_eval=False):
    """
    Creates a 2x length dataset where indexing is strictly gated by historical trends.
    Generates BOTH Global [B, 1] and Local [B, N] labels using Top-K and Fee Thresholds.
    """
    L, N, _ = features.shape
    testl = int(L * testl)
    validl = int(L * validl)

    if whale_params is None:
        pass # Placeholder for your whale injection logic
        
    # 1. Labeling Parameters from config
    x_h, x_f = cfg.x_h, cfg.x_f
    y_h, y_f = cfg.y_h, cfg.y_f
    z_h, z_f = cfg.z_h, cfg.z_f
    suffix = cfg.get_suffix()

    embargo = seq_len + z_f 

    # --- PHASE 1: GENERATE SYNTHETIC DATA ---
    injected_features = features # Replace with inject_whale_anomalies if used

    def get_trend_mask(data, span, local_z=2.0, breach_threshold=0.15, direction='future'):
        prices = data.squeeze(-1)
        T, N_stocks = prices.shape
        
        # 1. Calculate Returns
        shifts = np.zeros_like(prices)
        if direction == 'future':
            shifts[:-span] = prices[span:]
            with np.errstate(divide='ignore', invalid='ignore'):
                returns = (shifts - prices) / (prices + 1e-9)
            returns[-span:] = 0 
        else:
            shifts[span:] = prices[:-span]
            with np.errstate(divide='ignore', invalid='ignore'):
                returns = (prices - shifts) / (prices + 1e-9)
            returns[:span] = 0

        # --- LEVEL 1: GLOBAL ANOMALY (Historical Forecast) ---
        market_returns = np.mean(returns, axis=1)
        df_market = pd.Series(market_returns)

        forecast_mu = df_market.ewm(span=20, adjust=False).mean().shift(1).fillna(0).values
        variance = (df_market - forecast_mu)**2
        forecast_sigma = np.sqrt(variance.ewm(span=20, adjust=False).mean().shift(1).fillna(1e-4).values)
        
        forecast_mu_exp = np.expand_dims(forecast_mu, axis=1)
        forecast_sigma_exp = np.expand_dims(forecast_sigma, axis=1)
        
        upper_bound = forecast_mu_exp + (local_z * forecast_sigma_exp)
        lower_bound = forecast_mu_exp - (local_z * forecast_sigma_exp)
        
        breaches = (returns > upper_bound) | (returns < lower_bound)
        breach_ratio = np.sum(breaches, axis=1) / N_stocks
        global_mask = breach_ratio > breach_threshold
        
        # --- LEVEL 2: LOCAL ANOMALY (Top 5 + Transaction Fee Threshold) ---
        cross_mu = np.expand_dims(np.mean(returns, axis=1), axis=1)
        cross_sigma = np.expand_dims(np.std(returns, axis=1) + 1e-8, axis=1)
        local_z_scores = (returns - cross_mu) / cross_sigma
        
        abs_z_scores = np.abs(local_z_scores)
        abs_returns = np.abs(returns)
        
        # Initialize an empty boolean mask
        local_mask = np.zeros_like(returns, dtype=bool)
        
        # Step A: Find the indices of the Top 5 absolute z-scores for each day
        top_k = min(N_stocks // 10, N_stocks) # Safety check in case N < 5
        # argsort sorts ascending, so we take the last 'top_k' elements per row
        top_k_indices = np.argsort(abs_z_scores, axis=1)[:, -top_k:] 
        
        # Set the Top 5 positions to True using advanced indexing
        row_indices = np.arange(T)[:, None]
        local_mask[row_indices, top_k_indices] = True
        
        # Step B: Filter out anything that doesn't beat the transaction fee (1e-4)
        fee_threshold = 1e-4
        local_mask = local_mask & (abs_returns > fee_threshold)

        return global_mask, local_mask

    # Synthetic & Raw block masks
    F_injected_global, F_injected_local = get_trend_mask(injected_features, z_f, y_f, x_f, 'future')
    H_injected_global, _ = get_trend_mask(injected_features, z_h, y_h, x_h, 'history')
    
    F_raw_global, F_raw_local = get_trend_mask(features, z_f, y_f, x_f, 'future')
    H_raw_global, _ = get_trend_mask(features, z_h, y_h, x_h, 'history')

    # --- PHASE 3: CONCATENATE TO 2x LENGTH ---
    extended_features = np.concatenate([injected_features, features], axis=0)
    extended_H_masks = np.concatenate([H_injected_global, H_raw_global], axis=0)
    
    extended_F_global = np.concatenate([F_injected_global, F_raw_global], axis=0).astype(np.float32).reshape(-1, 1)
    extended_F_local = np.concatenate([F_injected_local, F_raw_local], axis=0).astype(np.float32) 

    # --- PHASE 4: FIT SCALER ON SYNTHETIC HALF ONLY ---
    train_scaler_data = extended_features[:L - z_f] 
    train_mean = np.mean(train_scaler_data)
    train_std = np.std(train_scaler_data) + 1e-8
    scaled_extended_features = (extended_features - train_mean) / train_std

    # --- PHASE 5: SMART INDEX ROUTING & AUGMENTATION ---
    def filter_and_balance(start_idx, end_idx, augment=True):
        pos_pool, neg_pool = [], []
        
        for i in range(start_idx, end_idx):
            T = i + seq_len 
            
            has_global_anomaly = (extended_F_global[T][0] > 0)
            has_history_anomaly = (extended_H_masks[T] > 0)
            
            if has_global_anomaly:
                pos_pool.append(i) 
            elif has_history_anomaly or (augment == False):
                neg_pool.append(i)
        print("pos_pool and neg_pool size ", len(pos_pool),len(neg_pool))
        if not pos_pool or not neg_pool:
            print(f"Warning: Missing classes in range {start_idx}-{end_idx}.")
            return [(i, i+seq_len) for i in (pos_pool + neg_pool)]
        
        if augment:
            target_size = min(len(pos_pool), len(neg_pool))
            pos_balanced = np.random.choice(pos_pool, target_size, replace=True).tolist()
            neg_balanced = np.random.choice(neg_pool, target_size, replace=True).tolist()
        else:
            pos_balanced = pos_pool
            neg_balanced = neg_pool

        balanced_indices = pos_balanced + neg_balanced
        np.random.shuffle(balanced_indices) 
        
        return [(i, i+seq_len) for i in balanced_indices]

    idx = {'train': [], 'valid': [], 'test': []}

    train_end_start_idx = L - seq_len - z_f - testl - embargo - validl
    idx['train'] = filter_and_balance(0, train_end_start_idx, augment=True)

    test_start = (2 * L) - testl - z_f - seq_len
    idx['test'] = filter_and_balance(test_start, test_start + testl, augment=balance_eval)

    valid_start = test_start - embargo - validl
    idx['valid'] = filter_and_balance(valid_start, valid_start + validl, augment=balance_eval)

    # --- PHASE 6: SAVE FILES ---
    os.makedirs(save_path, exist_ok=True)
    
    with open(f'{save_path}/data_anomaly{suffix}.pkl', 'wb') as f:
        pickle.dump({'processed_data': scaled_extended_features}, f)

    target_dict = {
        'global': extended_F_global,
        'local': extended_F_local
    }

    with open(f'{save_path}/label_anomaly{suffix}.pkl', 'wb') as f:
        pickle.dump({'processed_data': target_dict}, f) 

    with open(f'{save_path}/index_anomaly{suffix}.pkl', 'wb') as f:
        pickle.dump(idx, f)

    scaler = {
        'func': 're_standard_transform',
        'args': {'mean': float(train_mean), 'std': float(train_std)}
    }
    with open(f'{save_path}/scaler_anomaly{suffix}.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\n--- Output Complete ---")
    print(f"Global Target Shape: {extended_F_global.shape}")
    print(f"Local Target Shape: {extended_F_local.shape}")
    print(f"Train samples (Balanced Temporal Windows): {len(idx['train'])}")
    
    # Calculate the exact local imbalance ratio for the loss function
    total_local_elements = len(idx['train']) * N
    total_local_positives = sum(np.sum(extended_F_local[i+seq_len]) for i, _ in idx['train'])
    print(f"Actual Local Anomaly Ratio inside Balanced Train Windows: {total_local_positives / total_local_elements:.4f}")

# def generate_enhanced_anomaly_datasets(features, save_path, seq_len=12, testl=500, validl=500, whale_params=None, balance_eval=False):
#     """
#     Creates a 2x length dataset where indexing is strictly gated by historical trends.
#     Generates BOTH Global [B, 1] and Local [B, N] labels.
#     """
#     L, N, _ = features.shape
#     testl = int(L * testl)
#     validl = int(L * validl)

#     if whale_params is None:
#         NotImplementedError
        
#     # 1. Labeling Parameters from config
#     x_h, x_f = cfg.x_h, cfg.x_f
#     y_h, y_f = cfg.y_h, cfg.y_f
#     z_h, z_f = cfg.z_h, cfg.z_f
#     suffix = cfg.get_suffix()

#     embargo = seq_len + z_f 

#     # --- PHASE 1: GENERATE SYNTHETIC DATA ---
#     # print("Injecting whale anomalies into synthetic training block...")
#     # Assumes inject_whale_anomalies is defined in your environment
#     # injected_features = inject_whale_anomalies(features, **whale_params)
#     injected_features = features
#     def get_trend_mask(data, span, local_z=2.0, breach_threshold=0.15, direction='future'):
#         """
#         data: shape [T, N] (Time, Stocks)
#         global_k: Multiplier for historical forecasted volatility range.
#         local_z: Multiplier for today's cross-sectional standard deviation.
#         breach_threshold: % of stocks that must breach the forecasted range to trigger Global Anomaly.
#         """
#         prices = data.squeeze(-1)
#         T, N = prices.shape
        
#         # 1. Calculate Returns based on direction
#         shifts = np.zeros_like(prices)
#         if direction == 'future':
#             shifts[:-span] = prices[span:]
#             with np.errstate(divide='ignore', invalid='ignore'):
#                 returns = (shifts - prices) / prices
#             returns[-span:] = 0 
#         else: # 'history'
#             shifts[span:] = prices[:-span]
#             with np.errstate(divide='ignore', invalid='ignore'):
#                 returns = (prices - shifts) / shifts
#             returns[:span] = 0     

#         # --- LEVEL 1: GLOBAL ANOMALY (Historical Forecast) ---
#         # Calculate the average market return for each day
#         market_returns = np.mean(returns, axis=1)
        
#         # Use pandas EWMA for fast 1-step ahead forecasting based on history
#         # span=20 is roughly a 1-month trading history memory
#         df_market = pd.Series(market_returns)
#         forecast_mu = df_market.ewm(span=20, adjust=False).mean().shift(1).fillna(0).values
        
#         # Forecast Volatility (Variance) using EWMA
#         variance = (df_market - forecast_mu)**2
#         forecast_sigma = np.sqrt(variance.ewm(span=20, adjust=False).mean().shift(1).fillna(1e-4).values)
        
#         # Expand shapes to broadcast against individual stocks [T, N]
#         forecast_mu_exp = np.expand_dims(forecast_mu, axis=1)
#         forecast_sigma_exp = np.expand_dims(forecast_sigma, axis=1)
        
#         # Did the stock exceed the HISTORICALLY forecasted global range?
#         # (Checking if it's an MSE-style extreme outlier)

#         upper_bound = forecast_mu_exp + (local_z * forecast_sigma_exp)
#         lower_bound = forecast_mu_exp - (local_z * forecast_sigma_exp)
        
#         breaches = (returns > upper_bound) | (returns < lower_bound)
        
#         # If > X% of stocks breached the historically forecasted range, it's a global anomaly
#         breach_ratio = np.sum(breaches, axis=1) / N

#         global_mask = breach_ratio > breach_threshold
        
#         # --- LEVEL 2: LOCAL ANOMALY (Cross-Sectional Peers Today) ---
#         # Calculate today's center of mass and dispersion (MAE style)
#         cross_mu = np.expand_dims(np.mean(returns, axis=1), axis=1)
#         cross_sigma = np.expand_dims(np.std(returns, axis=1) + 1e-8, axis=1)

#         # Standardize stock's return based ONLY on what other stocks did today
#         local_z_scores = (returns - cross_mu) / cross_sigma
        
#         # Local Mask: Did this specific stock deviate from the pack today?
#         local_mask = np.abs(local_z_scores) > local_z
        
#         return global_mask, local_mask

#     # Synthetic block masks
#     F_injected_global, F_injected_local = get_trend_mask(injected_features, z_f, y_f, x_f, 'future')
#     H_injected_global, _ = get_trend_mask(injected_features, z_h, y_h, x_h, 'history')
    
#     # Raw block masks
#     F_raw_global, F_raw_local = get_trend_mask(features, z_f, y_f, x_f, 'future')
#     H_raw_global, _ = get_trend_mask(features, z_h, y_h, x_h, 'history')

#     # --- PHASE 3: CONCATENATE TO 2x LENGTH ---
#     extended_features = np.concatenate([injected_features, features], axis=0)
#     extended_H_masks = np.concatenate([H_injected_global, H_raw_global], axis=0)
    
#     # Format Targets
#     extended_F_global = np.concatenate([F_injected_global, F_raw_global], axis=0).astype(np.float32).reshape(-1, 1)
#     extended_F_local = np.concatenate([F_injected_local, F_raw_local], axis=0).astype(np.float32) # Shape: [2L, N]

#     # --- PHASE 4: FIT SCALER ON SYNTHETIC HALF ONLY ---
#     train_scaler_data = extended_features[:L - z_f] 
#     train_mean = np.mean(train_scaler_data)
#     train_std = np.std(train_scaler_data) + 1e-8
#     scaled_extended_features = (extended_features - train_mean) / train_std

#     # --- PHASE 5: SMART INDEX ROUTING & AUGMENTATION ---
#     def filter_and_balance(start_idx, end_idx, augment=True):
#         pos_pool, neg_pool = [], []
        
#         for i in range(start_idx, end_idx):
#             T = i + seq_len 
            
#             if extended_H_masks[T]:
#                 # We balance based on the GLOBAL trend to ensure equal regime exposure
#                 if extended_F_global[T][0] == 1.0:
#                     pos_pool.append(i) 
#                 else:
#                     neg_pool.append(i) 
                    
#             elif augment == False:
#                 neg_pool.append(i)

#         if not pos_pool or not neg_pool:
#             print(f"Warning: Missing classes in range {start_idx}-{end_idx}. Proceeding without balancing.")
#             return [(i, i+seq_len) for i in (pos_pool + neg_pool)]

#         if augment:
#             target_size = min(len(pos_pool), len(neg_pool))
#             pos_balanced = np.random.choice(pos_pool, target_size, replace=True).tolist()
#             neg_balanced = np.random.choice(neg_pool, target_size, replace=True).tolist()
#         else:
#             pos_balanced = pos_pool
#             neg_balanced = neg_pool

#         balanced_indices = pos_balanced + neg_balanced
#         np.random.shuffle(balanced_indices) 
        
#         return [(i, i+seq_len) for i in balanced_indices]

#     idx = {'train': [], 'valid': [], 'test': []}

#     train_end_start_idx = L - seq_len - z_f - testl - embargo - validl
#     idx['train'] = filter_and_balance(0, train_end_start_idx, augment=True)

#     test_start = (2 * L) - testl - z_f - seq_len
#     idx['test'] = filter_and_balance(test_start, test_start + testl, augment=balance_eval)

#     valid_start = test_start - embargo - validl

#     if valid_start < L:
#          raise ValueError(f"validl/testl bleed into synthetic data at {valid_start}!")
#     idx['valid'] = filter_and_balance(valid_start, valid_start + validl, augment=balance_eval)

#     # --- PHASE 6: SAVE FILES ---
#     os.makedirs(save_path, exist_ok=True)
    
#     with open(f'{save_path}/data_anomaly' + suffix + '.pkl', 'wb') as f:
#         pickle.dump({'processed_data': scaled_extended_features}, f)

#     # MERGE INTO DICTIONARY HERE
#     target_dict = {
#         'global': extended_F_global, # [2L, 1]
#         'local': extended_F_local    # [2L, N]
#     }

#     with open(f'{save_path}/label_anomaly' + suffix + '.pkl', 'wb') as f:
#         pickle.dump({'processed_data': target_dict}, f) 

#     with open(f'{save_path}/index_anomaly' + suffix + '.pkl', 'wb') as f:
#         pickle.dump(idx, f)

#     scaler = {
#         'func': 're_standard_transform',
#         'args': {'mean': float(train_mean), 'std': float(train_std)}
#     }
#     with open(f'{save_path}/scaler_anomaly' + suffix + '.pkl', 'wb') as f:
#         pickle.dump(scaler, f)

#     print(f"\n--- Output Complete ---")
#     print(f"Global Target Shape: {extended_F_global.shape}")
#     print(f"Local Target Shape: {extended_F_local.shape}")
#     print(f"Train samples (1:1 balanced): {len(idx['train'])}")
#     print(f"Train pos local: {F_injected_local.sum() / (F_injected_local.shape[0] * F_injected_local.shape[1])}")


@dataclass
class ParametersForAD:
    # 1. Dataset/Sequence Params
    seq_len: int = 24
    num_anomalies: int = 0
    
    # 只影响训练集
    x_h: float = 0.3
    y_h: float = 1
    z_h: int = 1
    
    # 影响测试结果
    x_f: float = 0.3
    y_f: float = 4
    z_f: int = 1

    # # 2. History Thresholds (his)
    # x_h: float = 0
    # y_h: float = 0
    # z_h: int = 1
    
    # # # 3. Future Thresholds (fut)
    # x_f: float = 0
    # y_f: float = 0
    # z_f: int = 1

    def get_suffix(self):
        """Generates the standardized filename suffix."""
        return f"_{self.seq_len}_{self.num_anomalies}_his{self.x_h}_{self.y_h}_{self.z_h}_fut{self.x_f}_{self.y_f}_{self.z_f}_"

cfg = ParametersForAD()
np.random.seed(42)
suffix = ''
suffix = '_300'
name = "Minute_Origin_data"
save_path2 = name + 'A'  + suffix
save_path = 'raw_data'
custom_cache_name = f"Minute_Origin_data_multivariate_aligned" + suffix + '.pkl'

# custom_cache_name = f"{name}_multivariate_aligned_300.pkl"
# if "csi500" in name:
#     save_path = f'raw_data/{name}/'
#     stock_list = (
#             ts_pro.index_weight(
#                 index_code="000905.SH").iloc[:500]["con_code"].tolist()
#     )
#     print("Using the constituent stocks of market index CSI500.")
# else:
#     save_path = f'/newhome/wbyan/stock_data/CH/{name}'
#     stock_list = (
#             ts_pro.index_weight(
#                 index_code="399300.SZ").iloc[:50]["con_code"].tolist()
#     )
#     print("Using the constituent stocks of market index CSI300.")

if __name__ == "__main__":
    # Create a unique cache name based on the dataset being loaded
    
    features = create_stock_data_numpy(save_path, cache_filename=custom_cache_name)
    # if features.shape[0] > 0:
    #         # 1. Calculate the mean, EXCLUDING zeros.
    #         # We make a temporary copy of day 0 where 0s are turned to NaNs, 
    #         # so nanmean() will correctly ignore them.
    #         temp_first_day = np.where(features[0] == 0, np.nan, features[0])
    #         first_day_means = np.nanmean(temp_first_day, axis=0)
            
    #         # Catch edge case: if ALL stocks are 0 for a specific feature, nanmean returns NaN.
    #         first_day_means = np.nan_to_num(first_day_means, nan=1e-6)
            
    #         # 2. Replace 0s on the first day with the calculated averages.
    #         # (Using features[0] == 0 instead of np.iszero)
    #         features[0] = np.where(features[0] == 0, first_day_means, features[0])
            
    #         # 3. Forward Fill: Iterate through time, replacing 0s with yesterday's price
    #         for t in range(1, features.shape[0]):
    #             features[t] = np.where(features[t] == 0, features[t-1], features[t])

    generate_enhanced_anomaly_datasets(
        features, 
        save_path2, 
        seq_len=cfg.seq_len, 
        testl=0.1,
        validl=0.1,
        whale_params={
            'num_anomalies': cfg.num_anomalies, 
            'z_window': 2, # or z_window2
            'intensity_range': (-0.1, 0.1), # From -15% to +15%
            'stock_ratio_range': (0.05, 0.15),   # Whale controls 10% to 20% of stocks
        }
    )