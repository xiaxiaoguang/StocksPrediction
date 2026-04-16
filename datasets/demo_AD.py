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
import itertools
from dataclasses import dataclass
from jqdatasdk import *
import tushare as ts
from tqdm import tqdm
import joblib
ts_pro = ts.pro_api("4883ec2948bef5ea9b8deb81b0c072b1808fb2b1c9d59f0d676a6095")

# try:
#     auth('18801046792','CIL@pku2114')
#     jq_state = True
# except Exception as e:
#     print(f'jq登录失败, 原因:{e}')

def create_stock_data_numpy(save_folder_path, stock_list=None):
    """
    Returns original closing prices in shape (L, N, 1) as a numpy array.
    L: Length, N: Number of Stocks, 1: Single Feature (Price)
    Matches stock_list (e.g., 000001.SZ) to files (e.g., 000001.XSHE.pkl) using 6-digit prefix.
    """
    # 1. Get all available files in the directory
    available_files = [f for f in os.listdir(save_folder_path) if f.endswith('.pkl')]
    
    # 2. Create a mapping of { '6_digit_prefix': 'full_filename.pkl' }
    # This handles the suffix difference (XSHE vs SZ)
    file_prefix_map = {f.split('.')[0]: f for f in available_files}

    if stock_list is None:
        # If no list provided, just use all available files
        matched_files = available_files
    else:
        # Extract prefixes from your CSI300 list (e.g., "000001.SZ" -> "000001")
        requested_prefixes = [s.split('.')[0] for s in stock_list]
        # Find which requested stocks actually exist in the minute-level folder
        matched_files = []
        for pref in requested_prefixes:
            if pref in file_prefix_map:
                matched_files.append(file_prefix_map[pref])
            else:
                # Optional: print(f"Warning: Stock prefix {pref} not found in minute data.")
                pass

    all_series = []
    valid_stocks = []
    # breakpoint()
    for file_name in tqdm(matched_files):
        file_path = os.path.join(save_folder_path, file_name)
        stock_name = file_name.replace('.pkl', '')
        try:
            df = joblib.load(file_path)
            if 'close' in df.columns:
                # Ensure the index is datetime for proper alignment during pd.concat
                price_series = df['close']
                all_series.append(price_series)
                valid_stocks.append(stock_name)
        except Exception as e:
            print(f"Error loading {file_name}: {e}")

    if not all_series:
        print("No valid data found.")
        return None, []

    combined_df = pd.concat(all_series, axis=1, keys=valid_stocks)
    combined_df = combined_df.ffill().fillna(0)
    data_bnl = combined_df.values[:, :, np.newaxis]
    print(f"Data prepared. Shape: {data_bnl.shape} (Timesteps, Stocks, Features)") 
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
    Generates BOTH Global [B, 1] and Local [B, N] labels using EWMA Forecasting & Cross-Sectional Analysis.
    """
    L, N, _ = features.shape
    testl = int(L * testl)
    validl = int(L * validl)
    if whale_params is None:
        raise NotImplementedError("whale_params must be provided")
        
    # 1. Labeling Parameters from config
    x_h, x_f = cfg.x_h, cfg.x_f  # Represents the 'breach_threshold' ratio (e.g., 0.15)
    y_h, y_f = cfg.y_h, cfg.y_f  # Represents the 'k' and 'z' volatility multipliers (e.g., 2.5)
    z_h, z_f = cfg.z_h, cfg.z_f  # Represents the lookahead/lookback spans
    suffix = cfg.get_suffix()

    embargo = seq_len + z_f 

    # --- PHASE 1: GENERATE SYNTHETIC DATA ---
    print("Injecting whale anomalies into synthetic training block...")
    # Assumes inject_whale_anomalies is defined in your environment
    injected_features = inject_whale_anomalies(features, **whale_params)

    def get_trend_mask(data, span, global_k=2.5, local_z=2.0, breach_threshold=0.15, direction='future'):
        """
        data: shape [T, N] (Time, Stocks)
        global_k: Multiplier for historical forecasted volatility range (MSE style).
        local_z: Multiplier for today's cross-sectional standard deviation (MAE style).
        breach_threshold: % of stocks that must breach the forecasted range to trigger Global Anomaly.
        """
        prices = data.squeeze(-1)
        T, N_stocks = prices.shape
        
        # 1. Calculate Returns based on direction
        shifts = np.zeros_like(prices)
        if direction == 'future':
            shifts[:-span] = prices[span:]
            with np.errstate(divide='ignore', invalid='ignore'):
                returns = (shifts - prices) / prices
            returns[-span:] = 0 
        else: # 'history'
            shifts[span:] = prices[:-span]
            with np.errstate(divide='ignore', invalid='ignore'):
                returns = (prices - shifts) / shifts
            returns[:span] = 0 
            
        returns[~np.isfinite(returns)] = 0
        
        # --- LEVEL 1: GLOBAL ANOMALY (Historical Forecast) ---
        # Calculate the average market return for each day
        market_returns = np.mean(returns, axis=1)
        
        # Use pandas EWMA for fast 1-step ahead forecasting based on history
        df_market = pd.Series(market_returns)
        forecast_mu = df_market.ewm(span=20, adjust=False).mean().shift(1).fillna(0).values
        
        # Forecast Volatility (Variance) using EWMA
        variance = (df_market - forecast_mu)**2
        forecast_sigma = np.sqrt(variance.ewm(span=20, adjust=False).mean().shift(1).fillna(1e-4).values)
        
        # Expand shapes to broadcast against individual stocks [T, N]
        forecast_mu_exp = np.expand_dims(forecast_mu, axis=1)
        forecast_sigma_exp = np.expand_dims(forecast_sigma, axis=1)
        
        # Did the stock exceed the HISTORICALLY forecasted global range?
        upper_bound = forecast_mu_exp + (global_k * forecast_sigma_exp)
        lower_bound = forecast_mu_exp - (global_k * forecast_sigma_exp)
        
        breaches = (returns > upper_bound) | (returns < lower_bound)
        
        # If > X% of stocks breached the historically forecasted range, it's a global anomaly
        breach_ratio = np.sum(breaches, axis=1) / N_stocks
        global_mask = breach_ratio > breach_threshold
        
        # --- LEVEL 2: LOCAL ANOMALY (Cross-Sectional Peers Today) ---
        # Calculate today's center of mass and dispersion
        cross_mu = np.expand_dims(np.mean(returns, axis=1), axis=1)
        cross_sigma = np.expand_dims(np.std(returns, axis=1) + 1e-8, axis=1)
        
        # Standardize stock's return based ONLY on what other stocks did today
        local_z_scores = (returns - cross_mu) / cross_sigma
        
        # Local Mask: Did this specific stock deviate from the pack today?
        local_mask = np.abs(local_z_scores) > local_z
        
        return global_mask, local_mask
        
    # --- PHASE 2: GENERATE MASKS (Using explicit kwargs to prevent mismatches) ---
    F_injected_global, F_injected_local = get_trend_mask(
        injected_features, span=z_f, global_k=y_f, local_z=y_f, breach_threshold=x_f, direction='future'
    )
    H_injected_global, _ = get_trend_mask(
        injected_features, span=z_h, global_k=y_h, local_z=y_h, breach_threshold=x_h, direction='history'
    )
    
    F_raw_global, F_raw_local = get_trend_mask(
        features, span=z_f, global_k=y_f, local_z=y_f, breach_threshold=x_f, direction='future'
    )
    H_raw_global, _ = get_trend_mask(
        features, span=z_h, global_k=y_h, local_z=y_h, breach_threshold=x_h, direction='history'
    )

    # --- PHASE 3: CONCATENATE TO 2x LENGTH ---
    extended_features = np.concatenate([injected_features, features], axis=0)
    extended_H_masks = np.concatenate([H_injected_global, H_raw_global], axis=0)
    
    # Format Targets
    extended_F_global = np.concatenate([F_injected_global, F_raw_global], axis=0).astype(np.float32).reshape(-1, 1)
    extended_F_local = np.concatenate([F_injected_local, F_raw_local], axis=0).astype(np.float32) # Shape: [2L, N]

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
            
            if extended_H_masks[T]:
                # We balance based on the GLOBAL trend to ensure equal regime exposure
                if extended_F_global[T][0] == 1.0:
                    pos_pool.append(i) 
                else:
                    neg_pool.append(i) 

        if not pos_pool or not neg_pool:
            print(f"Warning: Missing classes in range {start_idx}-{end_idx}. Proceeding without balancing.")
            return [(i, i+seq_len) for i in (pos_pool + neg_pool)]

        if augment:
            target_size = min(len(pos_pool), len(neg_pool))
            pos_balanced = np.random.choice(pos_pool, target_size, replace=True).tolist()
            neg_balanced = np.random.choice(neg_pool, target_size, replace=True).tolist()
            print(f"balanced to {target_size}")
        else:
            pos_balanced = pos_pool
            neg_balanced = neg_pool

        balanced_indices = pos_balanced + neg_balanced
        np.random.shuffle(balanced_indices) 
        
        return [(i, i+seq_len) for i in balanced_indices]

    idx = {'train': [], 'valid': [], 'test': []}

    train_end_start_idx = L - seq_len - z_f
    idx['train'] = filter_and_balance(0, train_end_start_idx, augment=True)

    test_start = (2 * L) - testl - z_f - seq_len
    idx['test'] = filter_and_balance(test_start, test_start + testl, augment=balance_eval)

    valid_start = test_start - embargo - validl
    if valid_start < L:
         raise ValueError(f"validl/testl bleed into synthetic data at {valid_start}!")
    idx['valid'] = filter_and_balance(valid_start, valid_start + validl, augment=balance_eval)
    print(f"range train {train_end_start_idx} test {test_start} valid {valid_start}")

    # --- PHASE 6: SAVE FILES ---
    os.makedirs(save_path, exist_ok=True)
    
    with open(f'{save_path}/data_anomaly' + suffix + '.pkl', 'wb') as f:
        pickle.dump({'processed_data': scaled_extended_features}, f)

    # MERGE INTO DICTIONARY HERE
    target_dict = {
        'global': extended_F_global, # [2L, 1]
        'local': extended_F_local    # [2L, N]
    }

    with open(f'{save_path}/label_anomaly' + suffix + '.pkl', 'wb') as f:
        pickle.dump({'processed_data': target_dict}, f) 

    with open(f'{save_path}/index_anomaly' + suffix + '.pkl', 'wb') as f:
        pickle.dump(idx, f)

    scaler = {
        'func': 're_standard_transform',
        'args': {'mean': float(train_mean), 'std': float(train_std)}
    }
    with open(f'{save_path}/scaler_anomaly' + suffix + '.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\n--- Output Complete ---")
    print(f"Global Target Shape: {extended_F_global.shape}")
    print(f"Local Target Shape: {extended_F_local.shape}")
    print(f"Train samples (1:1 balanced): {len(idx['train'])}")


# def generate_enhanced_anomaly_datasets1(features, save_path, seq_len=12, testl=500, validl=500, whale_params=None, balance_eval=False):
#     """
#     Creates a 2x length dataset where indexing is strictly gated by historical trends.
#     Generates BOTH Global [B, 1] and Local [B, N] labels.
#     """
#     L, N, _ = features.shape
    
#     if whale_params is None:
#         NotImplementedError
        
#     # 1. Labeling Parameters from config
#     x_h, x_f = cfg.x_h, cfg.x_f
#     y_h, y_f = cfg.y_h, cfg.y_f
#     z_h, z_f = cfg.z_h, cfg.z_f
#     suffix = cfg.get_suffix()

#     embargo = seq_len + z_f 

#     # --- PHASE 1: GENERATE SYNTHETIC DATA ---
#     print("Injecting whale anomalies into synthetic training block...")
#     # Assumes inject_whale_anomalies is defined in your environment
#     injected_features = inject_whale_anomalies(features, **whale_params)

#     # --- PHASE 2: CALCULATE HISTORICAL (H) AND FUTURE (F) TRENDS ---
#     # def get_trend_mask(data, span, y_val, x_val, direction='future'):
#     #     prices = data.squeeze(-1)
#     #     shifts = np.zeros_like(prices)
        
#     #     if direction == 'future':
#     #         shifts[:-span] = prices[span:]
#     #         with np.errstate(divide='ignore', invalid='ignore'):
#     #             pct = (shifts - prices) / prices
#     #             pct[~np.isfinite(pct)] = 0
#     #         pct[-span:] = 0 
#     #     else:
#     #         shifts[span:] = prices[:-span]
#     #         with np.errstate(divide='ignore', invalid='ignore'):
#     #             pct = (prices - shifts) / shifts
#     #             pct[~np.isfinite(pct)] = 0
#     #         pct[:span] = 0 
            
#     #     # LOCAL MASK: Which specific stocks triggered the anomaly? (Shape: [L, N])
#     #     local_mask = np.abs(pct) > y_val
        
#     #     # GLOBAL MASK: Did enough stocks trigger it to cross the threshold? (Shape: [L])
#     #     rapid_change_count = np.sum(local_mask, axis=1)
#     #     global_mask = (rapid_change_count / N) > x_val
        
#     #     return global_mask, local_mask
#     def get_trend_mask(data, span, global_k=2.5, local_z=2.0, breach_threshold=0.15, direction='future'):
#         """
#         data: shape [T, N] (Time, Stocks)
#         global_k: Multiplier for historical forecasted volatility range.
#         local_z: Multiplier for today's cross-sectional standard deviation.
#         breach_threshold: % of stocks that must breach the forecasted range to trigger Global Anomaly.
#         """
#         prices = data.squeeze(-1)
#         T, N = prices.shape
        
#         # 1. Calculate Returns (Assuming 'future' direction for label generation)
#         shifts = np.zeros_like(prices)
#         shifts[:-span] = prices[span:]
#         with np.errstate(divide='ignore', invalid='ignore'):
#             returns = (shifts - prices) / prices
#         returns[-span:] = 0 
#         returns[~np.isfinite(returns)] = 0
        
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
#         upper_bound = forecast_mu_exp + (global_k * forecast_sigma_exp)
#         lower_bound = forecast_mu_exp - (global_k * forecast_sigma_exp)
        
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

#     train_end_start_idx = L - seq_len - z_f
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


@dataclass
class ParametersForAD:
    # 1. Dataset/Sequence Params
    seq_len: int = 24
    num_anomalies: int = 1000
    
    # 2. History Thresholds (his)
    x_h: float = 0.2
    y_h: float = 2
    z_h: int = 1
    
    # 3. Future Thresholds (fut)
    x_f: float = 0.3
    y_f: float = 3
    z_f: int = 3

    # 2. History Thresholds (his)
    # x_h: float = 0
    # y_h: float = 0
    # z_h: int = 1
    
    # # 3. Future Thresholds (fut)
    # x_f: float = 0
    # y_f: float = 0
    # z_f: int = 1

    def get_suffix(self):
        """Generates the standardized filename suffix."""
        return f"_{self.seq_len}_{self.num_anomalies}_his{self.x_h}_{self.y_h}_{self.z_h}_fut{self.x_f}_{self.y_f}_{self.z_f}_"

cfg = ParametersForAD()


# name = "csi500"
name="Minute_Origin_data"
if "csi500" in name:
    save_path = f'raw_data/{name}/'
    stock_list = (
            ts_pro.index_weight(
                index_code="000905.SH").iloc[:500]["con_code"].tolist()
    )
    print("Using the constituent stocks of market index CSI500.")
else :
    save_path = f'/newhome/wbyan/stock_data/CH/{name}'
    stock_list = (
            ts_pro.index_weight(
                index_code="399300.SZ").iloc[:50]["con_code"].tolist()
    )
    print("Using the constituent stocks of market index CSI300.")


if __name__ == "__main__":
    # features = create_stock_data_numpy(save_path, stock_list)
    with open("/home/benyan2023/workspace/STEP/STEP/datasets/Minute_Origin_dataA/data_anomaly_12_0_his0_0_1_fut0_0_1_.pkl", 'rb') as f:
        features = pickle.load(f)['processed_data']
    with open("/home/benyan2023/workspace/STEP/STEP/datasets/Minute_Origin_dataA/scaler_anomaly_12_0_his0_0_1_fut0_0_1_.pkl", 'rb') as f:
        label = pickle.load(f)['args']
    tl = features.shape[0] // 2
    features = features[:tl,:]
    features = features * label['std'] + label['mean']
    save_path2 = name + 'A'
    generate_enhanced_anomaly_datasets(
        features, 
        save_path2, 
        seq_len=cfg.seq_len, 
        testl=0.2,
        validl=0.1,
        whale_params={
            'num_anomalies': cfg.num_anomalies, 
            'z_window': 2, # or z_window2
            'intensity_range': (-0.1, 0.1), # From -15% to +15%
            'stock_ratio_range': (0.05, 0.15),   # Whale controls 10% to 20% of stocks
        }
    )