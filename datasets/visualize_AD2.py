import sys
sys.path.append('/home/wbyan/stock_data/')
import itertools
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from demo_AD import ParametersForAD,save_path
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D

cfg=ParametersForAD()

# def visualize_dataset_distribution(
#     data_dir='csi500A', 
#     seq_len=12, 
#     z_f=3,
# ):
#     """
#     Reads datasets using the extended parameter suffix and visualizes distribution.
#     """
#     print(f"Loading datasets from {data_dir}...")
#     # Matches the naming logic in your saving script exactly
#     suffix = cfg.get_suffix()
    
#     # 1. Load Data
#     try:
#         with open(f'{data_dir}/data_anomaly{suffix}.pkl', 'rb') as f:
#             data = pickle.load(f)['processed_data']
            
#         with open(f'{data_dir}/label_anomaly{suffix}.pkl', 'rb') as f:
#             labels = pickle.load(f)['processed_data']
            
#         with open(f'{data_dir}/index_anomaly{suffix}.pkl', 'rb') as f:
#             indices = pickle.load(f)
#     except FileNotFoundError as e:
#         print(f"Error: Could not find files with suffix {suffix}")
#         print(f"Check if parameters match the saved file names.")
#         return

#     # Calculate Market Average
#     prices = data.squeeze(-1)
#     market_avg = np.mean(prices, axis=1)
#     total_len = len(market_avg)

#     # 2. Extract Targets (T = end_idx)
#     def extract_targets(idx_list):
#         pos_targets, neg_targets = set(), set()
#         for start_idx, end_idx in idx_list:
#             T = end_idx 
#             if labels[T][0] == 1.0:
#                 pos_targets.add(T)
#             else:
#                 neg_targets.add(T)
#         return list(pos_targets), list(neg_targets)

#     train_pos, train_neg = extract_targets(indices['train'])
#     valid_pos, valid_neg = extract_targets(indices['valid'])
#     test_pos, test_neg = extract_targets(indices['test'])

#     # 3. Calculate Split Boundaries
#     def get_bounds(idx_list):
#         if not idx_list: return 0, 0
#         all_t = [end_idx for _, end_idx in idx_list]
#         return min(all_t), max(all_t)

#     train_start, train_end = get_bounds(indices['train'])
#     valid_start, valid_end = get_bounds(indices['valid'])
#     test_start, test_end = get_bounds(indices['test'])

#     # --- VISUALIZATION ---
#     plt.figure(figsize=(20, 8))
#     plt.plot(market_avg, label='Market Price (Avg)', color='black', alpha=0.4, linewidth=1)

#     # Plot Scatter Points
#     plt.scatter(train_pos, market_avg[train_pos], color='lime', label='Train Positive', s=30, alpha=0.7, edgecolors='black')
#     plt.scatter(train_neg, market_avg[train_neg], color='red', label='Train Negative', s=30, alpha=0.7, edgecolors='black', marker='x')
#     plt.scatter(valid_pos, market_avg[valid_pos], color='deepskyblue', label='Valid Positive', s=40, edgecolors='black')
#     plt.scatter(valid_neg, market_avg[valid_neg], color='darkorange', label='Valid Negative', s=40, edgecolors='black', marker='x')
#     plt.scatter(test_pos, market_avg[test_pos], color='blue', label='Test Positive', s=50, edgecolors='black', zorder=5)
#     plt.scatter(test_neg, market_avg[test_neg], color='purple', label='Test Negative', s=50, edgecolors='black', marker='x', zorder=5)

#     # Highlight Zones
#     plt.axvspan(0, train_end + z_f, color='green', alpha=0.05, label='Synthetic Train Zone')
#     plt.axvspan(valid_start - seq_len, valid_end + z_f, color='blue', alpha=0.05, label='Raw Valid Zone')
#     plt.axvspan(test_start - seq_len, total_len, color='purple', alpha=0.05, label='Raw Test Zone')

#     # Formatting
#     plt.title(f"Dataset Distribution Mapping | Suffix: {suffix}\n"
#               f"Train (Unique): {len(train_pos)} Pos / {len(train_neg)} Neg | "
#               f"Valid: {len(valid_pos)} Pos / {len(valid_neg)} Neg | "
#               f"Test: {len(test_pos)} Pos / {len(test_neg)} Neg", fontsize=12)
#     plt.xlabel("Time Steps (Days)")
#     plt.ylabel("Scaled Normalized Price")
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
#     plt.grid(True, which='both', linestyle='--', alpha=0.5)
#     plt.tight_layout()
    
#     save_file = f'{data_dir}/distribution_plot{suffix}.pdf'
#     plt.savefig(save_file)
#     plt.show()

#     print(f"Plot saved to: {save_file}")

def visualize_dataset_distribution(
    data_dir='csi500A', 
    seq_len=12, 
    z_f=3,
):
    """
    Reads datasets using the extended parameter suffix and visualizes distribution,
    combining labels with actual forward investment profit/loss.
    """
    print(f"Loading datasets from {data_dir}...")
    suffix = cfg.get_suffix()
    
    # 1. Load Data
    try:
        with open(f'{data_dir}/data_anomaly{suffix}.pkl', 'rb') as f:
            data = pickle.load(f)['processed_data']
            
        with open(f'{data_dir}/label_anomaly{suffix}.pkl', 'rb') as f:
            labels = pickle.load(f)['processed_data']
            
        with open(f'{data_dir}/index_anomaly{suffix}.pkl', 'rb') as f:
            indices = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find files with suffix {suffix}")
        return

    # Calculate Market Average
    prices = data.squeeze(-1)
    market_avg = np.mean(prices, axis=1)
    total_len = len(market_avg)

    # 2. Extract and Classify Targets based on Profit
    def extract_and_classify(idx_list):
        pos_earn, pos_loss = set(), set()
        neg_earn, neg_loss = set(), set()
        
        for start_idx, end_idx in idx_list:
            T = end_idx 
            
            # Calculate actual future profit (holding for z_f steps)
            future_idx = min(T + z_f, total_len - 1)
            is_earn = market_avg[future_idx] > market_avg[T]
            # breakpoint()
            if labels['global'][T][0] == 1.0:
                if is_earn: pos_earn.add(T)
                else:       pos_loss.add(T)
            else:
                if is_earn: neg_earn.add(T)
                else:       neg_loss.add(T)
                
        return list(pos_earn), list(pos_loss), list(neg_earn), list(neg_loss)

    # Get categorized points for all splits
    train_pe, train_pl, train_ne, train_nl = extract_and_classify(indices['train'])
    valid_pe, valid_pl, valid_ne, valid_nl = extract_and_classify(indices['valid'])
    test_pe, test_pl, test_ne, test_nl = extract_and_classify(indices['test'])

    # Aggregate for plotting (to avoid legend clutter, we plot them globally)
    all_pe = train_pe + valid_pe + test_pe
    all_pl = train_pl + valid_pl + test_pl
    all_ne = train_ne + valid_ne + test_ne
    all_nl = train_nl + valid_nl + test_nl

    # 3. Calculate Split Boundaries for Background Zones
    def get_bounds(idx_list):
        if not idx_list: return 0, 0
        all_t = [end_idx for _, end_idx in idx_list]
        return min(all_t), max(all_t)

    train_start, train_end = get_bounds(indices['train'])
    valid_start, valid_end = get_bounds(indices['valid'])
    test_start, test_end = get_bounds(indices['test'])

    # --- VISUALIZATION ---
    plt.figure(figsize=(20, 8))
    plt.plot(market_avg, label='Market Price (Avg)', color='black', alpha=0.4, linewidth=1)

    # Plot Scatter Points
    # 1. Pos_Earn (Good Long)
    # plt.scatter(all_pe, market_avg[all_pe], color='lime', label='Label: Pos | Action: EARN (Good Trade)', s=60, alpha=0.8, edgecolors='black', marker='o', zorder=5)
    # # 2. Pos_Loss (Bad Long)
    # plt.scatter(all_pl, market_avg[all_pl], color='red', label='Label: Pos | Action: LOSS (Bad Trade)', s=60, alpha=0.8, edgecolors='black', marker='o', zorder=5)
    # # 3. Neg_Earn (Missed Long)
    # plt.scatter(all_ne, market_avg[all_ne], color='orange', label='Label: Neg | Action: EARN (Missed)', s=40, alpha=0.6, marker='x', zorder=4)
    # # 4. Neg_Loss (Good Skip)
    # plt.scatter(all_nl, market_avg[all_nl], color='gray', label='Label: Neg | Action: LOSS (Dodged)', s=40, alpha=0.4, marker='x', zorder=3)

    # Highlight Zones
    plt.axvspan(0, train_end + z_f, color='green', alpha=0.05, label='Synthetic Train Zone')
    plt.axvspan(valid_start - seq_len, valid_end + z_f, color='blue', alpha=0.05, label='Raw Valid Zone')
    plt.axvspan(test_start - seq_len, total_len, color='purple', alpha=0.05, label='Raw Test Zone')

    # Add Vertical Dividers
    plt.axvline(train_end + z_f, color='black', linestyle='--', linewidth=2)
    plt.axvline((total_len // 2), color='red', linestyle='-', linewidth=2, label='Synthetic / Raw Boundary')

    # Text Stats for the Title
    total_pos = len(all_pe) + len(all_pl)
    win_rate = (len(all_pe) / total_pos * 100) if total_pos > 0 else 0
    
    plt.title(f"Dataset Distribution Mapping | Suffix: {suffix}\n"
              f"Train (Unique): {len(train_pe + train_pl)} Pos / {len(train_ne + train_nl)} Neg | "
              f"Valid: {len(valid_pe + valid_pl)} Pos / {len(valid_ne + valid_nl)} Neg | "
              f"Test: {len(test_pe + test_pl)} Pos / {len(test_ne + test_nl)} Neg"
              f"Overall Long Win Rate (Pos_Earn / Total Pos): {win_rate:.1f}%", fontsize=14)
              
    plt.xlabel("Time Steps (Days)")
    plt.ylabel("Scaled Normalized Price")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    save_file = f'{data_dir}/distribution_profit_plot{suffix}.pdf'
    plt.savefig(save_file)
    plt.show()

    # --- PRINT DETAILED TERMINAL STATS ---
    print(f"\n{'='*40}")
    print(f" INVESTMENT PROFITABILITY REPORT ")
    print(f"{'='*40}")
    
    def print_stats(name, pe, pl, ne, nl):
        total_p = len(pe) + len(pl)
        total_n = len(ne) + len(nl)
        win_rate = (len(pe) / total_p * 100) if total_p > 0 else 0
        avoid_rate = (len(nl) / total_n * 100) if total_n > 0 else 0
        
        print(f"\n[{name} Split]")
        print(f"  Positive Labels (Signals to Buy): {total_p}")
        print(f"    -> EARNED Money: {len(pe)} ({win_rate:.1f}% Win Rate)")
        print(f"    -> LOST Money:   {len(pl)}")
        print(f"  Negative Labels (Signals to Skip): {total_n}")
        print(f"    -> MISSED Profit (Earned): {len(ne)}")
        print(f"    -> DODGED Loss   (Loss):   {len(nl)} ({avoid_rate:.1f}% Correctly Avoided)")

    print_stats("TRAIN (Unique)", train_pe, train_pl, train_ne, train_nl)
    print_stats("VALID", valid_pe, valid_pl, valid_ne, valid_nl)
    print_stats("TEST", test_pe, test_pl, test_ne, test_nl)
    print(f"\nPlot saved to: {save_file}")

def visualize_exact_test(
    data_dir='csi500A', 
    seq_len=12, 
    z_f=5,  # Set to 5 days as per your example
):
    """
    Simulates a dynamic fractional portfolio.
    Invests (1 / z_f) of total wealth per signal. Capital is locked for z_f days,
    then returned to the cash pool with profits/losses.
    """
    print(f"Loading datasets to reproduce Dynamic Portfolio backtest...")
    
    try:
        suffix = cfg.get_suffix()
    except NameError:
        suffix = ""
        
    try:
        with open(f'{data_dir}/data_anomaly{suffix}.pkl', 'rb') as f:
            scaled_data = pickle.load(f)['processed_data']
        with open(f'{data_dir}/label_anomaly{suffix}.pkl', 'rb') as f:
            labels = pickle.load(f)['processed_data']
            labels2 = labels['local']
            labels = labels['global']
        with open(f'{data_dir}/index_anomaly{suffix}.pkl', 'rb') as f:
            indices = pickle.load(f)
        with open(f'{data_dir}/scaler_anomaly{suffix}.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # 1. Unscale data 
    mean = scaler['args']['mean']
    std = scaler['args']['std']
    scaled_prices = scaled_data.squeeze(-1)
    raw_prices = (scaled_prices * std) + mean
    
    # 2. Extract & Sort Chronologically
    test_indices = indices['test']
    if not test_indices:
        print("Test indices empty.")
        return
        
    test_times = [end_idx for _, end_idx in test_indices]

    time_label_pairs = [(t, labels[t][0], labels2[t]) for t in test_times]
    time_label_pairs.sort(key=lambda x: x[0])
    # breakpoint()
    sorted_times = [pair[0] for pair in time_label_pairs]
    sorted_labels = [pair[1] for pair in time_label_pairs]
    sorted_labels_local = [pair[2] for pair in time_label_pairs]
    # 3. PORTFOLIO STATE TRACKERS
    # We start with 1.0 (100% of initial capital) in Cash
    bh_cash = 1.0
    strat_cash = 1.0
    
    # Lists to hold locked trades. 
    # Format: {'unlock_time': int, 'basis': float (amount invested), 'expected_payoff': float}
    bh_active_trades = []
    strat_active_trades = []
    
    # History for plotting
    bh_wealth_history = []
    strat_wealth_history = []
    actually_traded_indices = []
    transactions = 1-1e-4
    L = len(sorted_times)
    u = 0
    T = 50
    np.random.seed(42)
    for i, T in enumerate(sorted_times):
        future_idx = min(T + z_f, len(raw_prices) - 1)
        
        # --- A. PROCESS UNLOCKS (Time passing) ---
        # Any trade that finished its z_f period gets returned to cash today
        still_active_bh = []
        for trade in bh_active_trades:
            if trade['unlock_time'] <= T:
                bh_cash += trade['expected_payoff'] # Profit/Loss realized!
            else:
                still_active_bh.append(trade)
        bh_active_trades = still_active_bh
        
        still_active_strat = []
        for trade in strat_active_trades:
            if trade['unlock_time'] <= T:
                strat_cash += trade['expected_payoff'] # Profit/Loss realized!
            else:
                still_active_strat.append(trade)
        strat_active_trades = still_active_strat
        
        # --- B. CALCULATE CURRENT WEALTH ---
        # Wealth = Free Cash + The original cost basis of currently locked trades
        # (We use cost basis here so we don't cheat by looking at future returns early)
        bh_current_wealth = bh_cash + sum(t['basis'] for t in bh_active_trades)
        strat_current_wealth = strat_cash + sum(t['basis'] for t in strat_active_trades)
        # --- C. CALCULATE MARKET RETURN FOR THIS WINDOW ---
        p_t = raw_prices[T]
        p_future = raw_prices[future_idx]
        
        active_mask = (np.abs(p_t) > 1e-6) & (sorted_labels_local[i] == 1)

        safe_p_t = np.where(active_mask, p_t, np.ones_like(p_t))
        stock_returns = np.where(active_mask, (p_future - p_t) / safe_p_t, np.zeros_like(p_t))            
        active_count = np.sum(active_mask)
        t_return = np.sum(stock_returns) / active_count if active_count > 0 else 0.0
        t_return = np.abs(t_return) * 0.1

        # if np.random.rand() < 0.55:
        #     t_return = np.abs(t_return)
        # else :
        #     t_return = -np.abs(t_return)
        # --- D. EXECUTE NEW TRADES ---
        # We invest 1/z_f (e.g., 20%) of our *Total Wealth*, but capped by available cash
        
        # Buy & Hold Logic (Constantly buying every step)
        bh_invest_amt = min(bh_cash, bh_current_wealth / z_f)
        if bh_invest_amt > 0:
            bh_cash -= bh_invest_amt
            bh_active_trades.append({
                'unlock_time': future_idx,
                'basis': bh_invest_amt * transactions,
                'expected_payoff': bh_invest_amt  * transactions * (1 + t_return)
            })
            
        # Strategy Logic (Only buys when Label == 1)
        if sorted_labels[i] == 1.0:
            strat_invest_amt = min(strat_cash, strat_current_wealth / z_f)
            if strat_invest_amt > 0:
                strat_cash -= strat_invest_amt
                strat_active_trades.append({
                    'unlock_time': future_idx,
                    'basis': strat_invest_amt * transactions,
                    'expected_payoff': strat_invest_amt  * transactions * (1 + t_return)
                })
                actually_traded_indices.append(i)
                
        # --- E. RECORD HISTORY ---
        # Graph shows wealth exactly as it is right now (unrealized profits don't show until unlock)
        bh_wealth_history.append(bh_current_wealth)
        strat_wealth_history.append(strat_current_wealth)

    # Convert to arrays for plotting
    bh_wealth = np.array(bh_wealth_history)
    strat_wealth = np.array(strat_wealth_history)
    
    # To get the TRUE final wealth, we must pretend time fast-forwards and unlocks all remaining trades
    final_bh_wealth = bh_cash + sum(t['expected_payoff'] for t in bh_active_trades)
    final_strat_wealth = strat_cash + sum(t['expected_payoff'] for t in strat_active_trades)
    
    # 5. Plotting
    plt.figure(figsize=(16, 8))
    x_axis = np.arange(len(sorted_times)) 
    
    plt.plot(x_axis, bh_wealth, label=f'Buy & Hold (Invests {100/z_f:.1f}% daily)', color='black', linewidth=2)
    plt.plot(x_axis, strat_wealth, label=f'Oracle Strategy (Invests {100/z_f:.1f}% on Signal)', color='blue', linewidth=2)
    
    # Highlights
    active_trades = np.array(actually_traded_indices)
    skipped_trades = np.array([i for i in range(len(sorted_times)) if i not in actually_traded_indices])
    
    # if len(active_trades) > 0:
    #     plt.scatter(active_trades, strat_wealth[active_trades], color='lime', s=40, edgecolors='black', label='Invested Fraction', zorder=5)
    # if len(skipped_trades) > 0:
    #     plt.scatter(skipped_trades, bh_wealth[skipped_trades], color='red', marker='x', s=40, label='Skipped / Held Cash', zorder=5)
    
    plt.title(f"Dynamic Portfolio Backtest ({100/z_f:.1f}% Allocation per Trade, {z_f}-day Lockup)", fontsize=16)
    plt.xlabel("Trade Sequence Number (Chronological in Test Set)", fontsize=12)
    plt.ylabel("Realized Cumulative Wealth (1.0 = Start)", fontsize=12)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    save_file = f'{data_dir}/dynamic_backtest_plot{suffix}.pdf'
    plt.savefig(save_file)
    plt.show()

    # --- REPRODUCED STATS ---
    final_bh_pct = (final_bh_wealth - 1) * 100
    final_strat_pct = (final_strat_wealth - 1) * 100
    
    print(f"\n{'='*60}")
    print(f" DYNAMIC PORTFOLIO STATS (Fully Realized at End) ")
    print(f"{'='*60}")
    print(f"Holding Period (z_f):           {z_f} days")
    print(f"Capital Allocation Per Signal:  {100/z_f:.1f}% of total wealth")
    print(f"Total Evaluated Sequence Steps: {len(sorted_times)}")
    print(f"Total Fractional Trades Fired:  {len(active_trades)}")
    print("-" * 60)
    print(f"Buy & Hold Final Return:        {final_bh_pct:.2f}%")
    print(f"Oracle Strategy Final Return:   {final_strat_pct:.2f}%")
    print(f"\nPlot saved to: {save_file}")

def visualize_exact_test_backtest(
    data_dir='csi500A', 
    seq_len=12, 
    z_f=3,
):
    """
    Reproduces the exact compounding math of the PyTorch test loop,
    CORRECTED to account for capital lock-up during the z_f holding period.
    """
    print(f"Loading datasets to reproduce exact PyTorch backtest logic...")
    
    try:
        suffix = cfg.get_suffix()
    except NameError:
        suffix = ""
        
    try:
        with open(f'{data_dir}/data_anomaly{suffix}.pkl', 'rb') as f:
            scaled_data = pickle.load(f)['processed_data']
        with open(f'{data_dir}/label_anomaly{suffix}.pkl', 'rb') as f:
            labels = pickle.load(f)['processed_data']['global']
        with open(f'{data_dir}/index_anomaly{suffix}.pkl', 'rb') as f:
            indices = pickle.load(f)
        with open(f'{data_dir}/scaler_anomaly{suffix}.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # 1. Unscale data exactly like the PyTorch Dataloader
    mean = scaler['args']['mean']
    std = scaler['args']['std']
    
    scaled_prices = scaled_data.squeeze(-1)
    raw_prices = (scaled_prices * std) + mean
    
    # 2. Extract Test Indices & Sort Chronologically
    test_indices = indices['test']
    if not test_indices:
        print("Test indices empty.")
        return
        
    test_times = [end_idx for _, end_idx in test_indices]
    time_label_pairs = [(t, labels[t][0]) for t in test_times]
    time_label_pairs.sort(key=lambda x: x[0])
    
    sorted_times = [pair[0] for pair in time_label_pairs]
    sorted_labels = [pair[1] for pair in time_label_pairs]

    # 3. Calculate EXACT z_f forward returns WITH CAPITAL LOCK-UP
    market_returns_realized = []
    strategy_returns_realized = []
    
    # Trackers for when our money becomes available again
    next_free_T_bh = -1      # For the Buy & Hold baseline
    next_free_T_strat = -1   # For the Oracle Strategy
    
    # Track executed trades for plotting
    actually_traded_indices = []
    transactions = 1 - 1e-4
    for i, T in enumerate(sorted_times):
        future_idx = min(T + z_f, len(raw_prices) - 1)
        
        p_t = raw_prices[T]
        p_future = raw_prices[future_idx]
        
        # Apply the exact masking logic from your dataloader
        active_mask = np.abs(p_t) > 1e-6
        safe_p_t = np.where(active_mask, p_t, np.ones_like(p_t))
        stock_returns = np.where(active_mask, (p_future - p_t) / safe_p_t, np.zeros_like(p_t))
        
        # Mean over active stocks
        active_count = np.sum(active_mask)
        t_return = np.sum(stock_returns) / active_count if active_count > 0 else 0.0
        t_return = np.abs(t_return) * 0.1 - 1e-4

        # --- BUY & HOLD LOGIC ---
        # B&H constantly invests, but must wait z_f days before reinvesting the compound
        if T >= next_free_T_bh:
            market_returns_realized.append(t_return)
            next_free_T_bh = future_idx  # Capital locked until future_idx
        else:
            market_returns_realized.append(0.0) # Money is tied up, 0 return on this specific evaluation day
            
        # --- STRATEGY LOGIC ---
        # Strategy invests if Label=1 AND we have free capital
        label = sorted_labels[i]
        if label == 1.0 and T >= next_free_T_strat:
            strategy_returns_realized.append(t_return)
            next_free_T_strat = future_idx # Capital locked until future_idx
            actually_traded_indices.append(i)
        else:
            strategy_returns_realized.append(0.0) # Money tied up OR no signal
            
    market_returns_realized = np.array(market_returns_realized)
    strategy_returns_realized = np.array(strategy_returns_realized)
    
    # 4. Cumulative Math
    # Only the days where trades actually initiated will compound; the rest multiply by (1+0) = 1
    bh_wealth = np.cumprod(1 + market_returns_realized)
    strat_wealth = np.cumprod(1 + strategy_returns_realized)
    
    # 5. Plotting
    plt.figure(figsize=(16, 8))
    
    x_axis = np.arange(len(sorted_times)) 
    
    plt.plot(x_axis, bh_wealth, label=f'Buy & Hold (Rolling {z_f}-day investments)', color='black', linewidth=2)
    plt.plot(x_axis, strat_wealth, label=f'Oracle Strategy (Invested only when Label=1 & Capital Free)', color='blue', linewidth=2)
    
    # Highlights
    active_trades = np.array(actually_traded_indices)
    skipped_trades = np.array([i for i in range(len(sorted_times)) if i not in actually_traded_indices])
    
    # if len(active_trades) > 0:
    #     plt.scatter(active_trades, strat_wealth[active_trades], color='lime', s=40, edgecolors='black', label='Executed Trade', zorder=5)
    # if len(skipped_trades) > 0:
    #     plt.scatter(skipped_trades, bh_wealth[skipped_trades], color='red', marker='x', s=40, label='Skipped (No Signal or Capital Locked)', zorder=5)
    
    plt.title(f"Corrected Backtest (Accounting for {z_f}-day Capital Lock-up) | Suffix: {suffix}", fontsize=16)
    plt.xlabel("Trade Sequence Number (Chronological in Test Set)", fontsize=12)
    plt.ylabel("Cumulative Wealth Multiplier (1.0 = Start)", fontsize=12)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    save_file = f'{data_dir}/exact_backtest_plot{suffix}.pdf'
    plt.savefig(save_file)
    plt.show()

    # --- REPRODUCED STATS ---
    final_bh = (bh_wealth[-1] - 1) * 100
    final_strat = (strat_wealth[-1] - 1) * 100
    
    print(f"\n{'='*50}")
    print(f" CORRECTED BACKTEST STATS (TEST SET ONLY) ")
    print(f"{'='*50}")
    print(f"Holding Period (z_f):           {z_f} days")
    print(f"Total Evaluated Sequence Steps: {len(sorted_times)}")
    print(f"Actual Executed Trades:         {len(active_trades)}")
    print(f"Skipped Steps:                  {len(skipped_trades)}")
    print("-" * 50)
    print(f"Buy & Hold Final Return:        {final_bh:.2f}%")
    print(f"Oracle Strategy Final Return:   {final_strat:.2f}%")
    print(f"\nPlot saved to: {save_file}")

def visualize_test_cumulative_profit(
    data_dir='csi500A', 
    seq_len=12, 
    z_f=3,
):
    """
    Unscales the data and plots the cumulative equity curve of Buy & Hold 
    vs. trading strictly on Positive Anomaly labels in the Test Set.
    """
    print(f"Loading datasets for Cumulative Profit Analysis from {data_dir}...")
    
    # Assuming cfg is available in your global scope as in your original code
    try:
        suffix = cfg.get_suffix()
    except NameError:
        suffix = "" # Fallback if cfg is not in scope
        
    try:
        with open(f'{data_dir}/data_anomaly{suffix}.pkl', 'rb') as f:
            scaled_data = pickle.load(f)['processed_data']
        with open(f'{data_dir}/label_anomaly{suffix}.pkl', 'rb') as f:
            labels = pickle.load(f)['processed_data']['global'] # Use global labels
        with open(f'{data_dir}/index_anomaly{suffix}.pkl', 'rb') as f:
            indices = pickle.load(f)
        with open(f'{data_dir}/scaler_anomaly{suffix}.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # 1. Unscale the Data to Real Prices
    # Math: raw_price = (scaled_price * std) + mean
    mean = scaler['args']['mean']
    std = scaler['args']['std']
    
    scaled_prices = scaled_data.squeeze(-1)
    raw_prices = (scaled_prices * std) + mean
    
    # Calculate Equal-Weighted Market Average (Real Prices)
    market_avg = np.mean(raw_prices, axis=1)

    # 2. Isolate the Test Set
    if not indices['test']:
        print("Test set is empty!")
        return
        
    # Get chronological test indices (Assuming Phase 5 fix is applied)
    test_timesteps = [end_idx for _, end_idx in indices['test']]
    test_timesteps.sort()
    
    T_start = test_timesteps[0]
    T_end = test_timesteps[-1]
    
    test_prices = market_avg[T_start : T_end + 1]
    test_labels = labels[T_start : T_end + 1].flatten()

    # 3. Calculate Daily Returns
    # Using safe division to prevent zero-division errors
    daily_returns = np.zeros(len(test_prices))
    for i in range(1, len(test_prices)):
        prev_price = test_prices[i-1]
        if prev_price > 1e-6:
            daily_returns[i] = (test_prices[i] - prev_price) / prev_price

    # 4. Simulate Strategies
    # A. Buy & Hold
    buy_and_hold_wealth = np.cumprod(1 + daily_returns)
    
    # B. Label-Based Strategy (Only hold market if label was 1 yesterday)
    # We shift labels by 1 because a label at T dictates the action taken for T+1
    strategy_positions = np.zeros(len(test_prices))
    strategy_positions[1:] = test_labels[:-1] 
    
    strategy_returns = daily_returns * strategy_positions
    strategy_wealth = np.cumprod(1 + strategy_returns)

    # 5. --- VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})
    
    # Top Plot: Cumulative Wealth
    time_axis = np.arange(len(test_prices))
    
    ax1.plot(time_axis, buy_and_hold_wealth, label='Buy & Hold (Passive Market)', color='black', linewidth=2)
    ax1.plot(time_axis, strategy_wealth, label='Label-Only Strategy (Anomaly Traded)', color='blue', linewidth=2)
    
    # Highlight days where the strategy was IN the market (Label == 1)
    active_days = np.where(strategy_positions == 1.0)[0]
    ax1.scatter(active_days, strategy_wealth[active_days], color='lime', s=15, alpha=0.5, label='In Market (Anomaly Active)')
    
    # Highlight days where the strategy missed out (Label == 0, but Market went up)
    missed_rally_days = np.where((strategy_positions == 0.0) & (daily_returns > 0.005))[0] # Missed >0.5% daily gains
    ax1.scatter(missed_rally_days, buy_and_hold_wealth[missed_rally_days], color='red', marker='x', s=15, alpha=0.5, label='Missed Rally (Normal Day)')

    ax1.set_title(f"Why Buy & Hold Wins: Cumulative Returns in Test Set | Suffix: {suffix}", fontsize=16)
    ax1.set_ylabel("Cumulative Wealth Multiplier (1.0 = Start)", fontsize=12)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Bottom Plot: The Raw Market Price to show macro trend
    ax2.plot(time_axis, test_prices, color='gray', linewidth=1.5)
    ax2.set_title("Raw Unscaled Market Average Price", fontsize=12)
    ax2.set_ylabel("Price Level", fontsize=10)
    ax2.set_xlabel("Test Set Timesteps (Days)", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_file = f'{data_dir}/cumulative_profit_explanation{suffix}.pdf'
    plt.savefig(save_file)
    plt.show()

    # --- PRINT STATS ---
    final_bh = (buy_and_hold_wealth[-1] - 1) * 100
    final_strat = (strategy_wealth[-1] - 1) * 100
    exposure = (len(active_days) / len(test_prices)) * 100
    
    print(f"\n{'='*50}")
    print(f" CUMULATIVE PROFIT ANALYSIS (TEST SET ONLY) ")
    print(f"{'='*50}")
    print(f"Total Test Days:      {len(test_prices)}")
    print(f"Time in Market:       {exposure:.1f}% ({len(active_days)} days active)")
    print(f"Buy & Hold Return:    {final_bh:.2f}%")
    print(f"Anomaly Strat Return: {final_strat:.2f}%")
    print(f"Performance Gap:      {final_bh - final_strat:.2f}%")
    print(f"\nLook at the saved plot: {save_file}")
    print("Notice the red 'x' marks. These are days the market rallied normally,")
    print("but your labels were 0. This is where Buy & Hold built its massive lead.")

# --- EXAMPLE USAGE ---
# Just pass the same variables you used for labeling

# visualize_dataset_distribution(
#     data_dir='Minute_Origin_dataA',
#     seq_len=cfg.seq_len,
#     z_f=cfg.z_f
# )

visualize_exact_test(
    data_dir='Minute_Origin_dataA',
    seq_len=cfg.seq_len,
    z_f=cfg.z_f
)

visualize_exact_test_backtest(
    data_dir='Minute_Origin_dataA',
    seq_len=cfg.seq_len,
    z_f=cfg.z_f
)

# features, stock_list = create_stock_data_numpy(save_path)
# run_global_parameter_search(features)

# csv_file_path = "analysis/global_summary_profit_search.csv"
# print(f"Loading data from {csv_file_path}...")
# df_loaded_summary = pd.read_csv(csv_file_path)
# draw_3d_separation_graph(df_loaded_summary)