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

def evaluate_investment_profitability(data_bnl, x_thresh, y_thresh, z_window):
    """
    Evaluates both anomaly prediction (volatility forecasting) and 
    investment profitability (directional forecasting).
    Returns a comprehensive dictionary of all calculated metrics.
    """
    L, N = data_bnl.shape[:2]
    prices = data_bnl.squeeze(-1)
    
    # Calculate market average for profit evaluation
    market_avg = np.mean(prices, axis=1)
    
    x_h, x_f = x_thresh
    y_h, y_f = y_thresh
    z_h, z_f = z_window

    def get_market_event_mask(data, span, y_val, x_val):
        shifts = np.zeros_like(data)
        shifts[:-span] = data[span:]
        with np.errstate(divide='ignore', invalid='ignore'):
            pct = (shifts - data) / data
            pct[~np.isfinite(pct)] = 0
        rapid_change_count = np.sum(np.abs(pct) > y_val, axis=1)
        return (rapid_change_count / N) > x_val

    # 1. Get raw event masks
    events_h_mask = get_market_event_mask(prices, z_h, y_h, x_h)
    events_f_mask = get_market_event_mask(prices, z_f, y_f, x_f)

    # 2. Setup counters over the valid pivot range
    valid_time_steps = range(z_h, L - z_f)
    
    # --- Original Anomaly Counters ---
    hist_count = 0
    fut_count = 0
    anomaly_count = 0 
    
    # --- New Profitability Counters ---
    pos_earn = 0
    pos_loss = 0
    neg_earn = 0
    neg_loss = 0
    
    # 3. Main Evaluation Loop
    for t in valid_time_steps:
        # History event ended exactly at T
        has_h = events_h_mask[t - z_h]
        # Future event starts exactly at T
        has_f = events_f_mask[t]
        
        # A. Update Anomaly/Volatility Logic
        if has_h: hist_count += 1
        if has_f: fut_count += 1
        if has_h and has_f: anomaly_count += 1

        # B. Update Profitability/Directional Logic (Holding for z_f steps)
        is_earn = market_avg[t + z_f] > market_avg[t]
        
        if has_h and has_f:  # Positive Signal -> Action: Long
            if is_earn: pos_earn += 1
            else:       pos_loss += 1
        elif has_h:      # Negative Signal -> Action: Skip
            if is_earn: neg_earn += 1
            else:       neg_loss += 1

    # 4. Calculate Final Metrics
    
    # Original: P(Future | History) and P(History | Future)
    prob_f_given_h = (anomaly_count / hist_count) if hist_count > 0 else 0.0
    prob_h_given_f = (anomaly_count / fut_count) if fut_count > 0 else 0.0
    
    # New: Win Rate and Avoid Rate
    total_pos = pos_earn + pos_loss
    total_neg = neg_earn + neg_loss
    win_rate = (pos_earn / total_pos) if total_pos > 0 else 0.0
    avoid_rate = (neg_loss / total_neg) if total_neg > 0 else 0.0

    # 5. Return Master Dictionary
    return {
        # Base Parameters
        "x_h": x_h, "x_f": x_f,
        "y_h": y_h, "y_f": y_f,
        "z_h": z_h, "z_f": z_f,
        
        # Original Anomaly / Volatility Stats
        "Hist_Events": hist_count,
        "Fut_Events": fut_count,
        "Anomalies": anomaly_count,
        "Signal_Strength_P(F|H)": round(prob_f_given_h, 4),
        "Recall_P(H|F)": round(prob_h_given_f, 4),
        
        # Profitability Stats
        "Total_Signals": total_pos,
        "Pos_Earn": pos_earn,
        "Pos_Loss": pos_loss,
        "Neg_Earn": neg_earn,
        "Neg_Loss": neg_loss,
        "Win_Rate(%)": round(win_rate * 100, 2),
        "Avoid_Rate(%)": round(avoid_rate * 100, 2)
    }
    
def evaluate_investment_in_memory(data_bnl, x_thresh, y_thresh, z_window):
    """
    Evaluates strategy profitability purely in memory using compounding math.
    Triggers a 'Long' position holding for z_f days when a historical event (has_h) is found.
    """
    L, N = data_bnl.shape[:2]
    prices = data_bnl.squeeze(-1)
    
    # Calculate market average for profit evaluation
    market_avg = np.mean(prices, axis=1)
    
    x_h, x_f = x_thresh
    y_h, y_f = y_thresh
    z_h, z_f = z_window

    def get_market_event_mask(data, span, y_val, x_val):
        shifts = np.zeros_like(data)
        shifts[:-span] = data[span:]
        with np.errstate(divide='ignore', invalid='ignore'):
            pct = (shifts - data) / data
            pct[~np.isfinite(pct)] = 0
        rapid_change_count = np.sum(np.abs(pct) > y_val, axis=1)
        return (rapid_change_count / N) > x_val

    # 1. Get raw historical event masks
    events_h_mask = get_market_event_mask(prices, z_h, y_h, x_h)

    # 2. Setup over the valid pivot range
    valid_time_steps = range(z_h, L - z_f)
    
    market_returns = []
    strategy_positions = []
    
    # Trackers for when our money becomes available again
    next_free_T_bh = -1      # For the Buy & Hold baseline
    next_free_T_strat = -1   # For the Oracle Strategy

    # 3. Main Evaluation Loop
    for t in valid_time_steps:
        future_idx = t + z_f

        # History event ended exactly at T
        has_h = events_h_mask[t - z_h]
        
        # Market return holding for z_f steps
        p_t = market_avg[t]
        p_future = market_avg[future_idx]
        
        # Safe division for percentage return
        if p_t > 1e-6:
            t_return = (p_future - p_t) / p_t
        else:
            t_return = 0.0

        if t >= next_free_T_bh:
            market_returns.append(t_return)
            next_free_T_bh = future_idx  # Capital locked until future_idx
        else:
            market_returns.append(0.0) # Money is tied up, 0 return on this specific evaluation day

        if has_h == 1.0 and t >= next_free_T_strat:
            strategy_positions.append(t_return)
            next_free_T_strat = future_idx # Capital locked until future_idx
        else:
            strategy_positions.append(0.0) # Money tied up OR no signal
    
    # 4. Calculate Final Compounding Metrics
    market_returns = np.array(market_returns)
    strategy_positions = np.array(strategy_positions)    

    strat_wealth = np.cumprod(1 + market_returns)
    bh_wealth = np.cumprod(1 + strategy_positions)

    final_strat_profit = (strat_wealth[-1] - 1) * 100 if len(strat_wealth) > 0 else 0.0
    final_bn_profit = (bh_wealth[-1] - 1) * 100 if len(bh_wealth) > 0 else 0.0

    return {
        "x1": x_h, "y1": y_h, "z1": z_h,
        "x_f": x_f, "y_f": y_f, "z_f": z_f,
        "Strategy_Profit(%)": final_strat_profit,
        "Strategy_Pos_label_Profit(%)": final_bn_profit
    }


def draw_3d_separation_graph(df):
    """
    Plots a 3D scatter of x1, y1, z1 colored by Strategy Profit (%),
    and calculates a separation plane for Profit > 0%.
    """
    print("\nDrawing 3D separation graph for Final Incomes Ratio...")
    
    # Drop NaNs
    df = df.dropna(subset=['Avg_Profit(%)'])
    
    X = df[['x1', 'y1', 'z1']].values
    y_profit = df['Avg_Profit(%)'].values
    
    # Binary classes: 1 if profitable (> 0%), 0 if loss-making (<= 0%)
    y_class = (y_profit > 0.0).astype(int)
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the scatter points. Red for loss, Green for profit
    scatter = ax.scatter(
        df['x1'], df['y1'], df['z1'], 
        c=y_profit, cmap='RdYlGn', 
        s=df['Avg_Total_Signals'] * 0.5 + 10, # Size based on signals (+10 baseline so small sizes remain visible)
        edgecolors='k', alpha=0.8
    )
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Average Final Income Ratio / Profit (%)')

    # Fit SVM to draw a separation plane
    if len(np.unique(y_class)) == 2:
        clf = SVC(kernel='linear', C=1.0)
        clf.fit(X, y_class)
        
        w = clf.coef_[0]
        b = clf.intercept_[0]
        
        xx, yy = np.meshgrid(
            np.linspace(df['x1'].min(), df['x1'].max(), 10),
            np.linspace(df['y1'].min(), df['y1'].max(), 10)
        )
        
        if w[2] != 0: 
            zz = (-w[0] * xx - w[1] * yy - b) / w[2]
            
            z_min, z_max = df['z1'].min(), df['z1'].max()
            zz = np.clip(zz, z_min, z_max)
            
            # Plot the plane
            ax.plot_surface(xx, yy, zz, color='blue', alpha=0.2, rstride=100, cstride=100)
            print("Separation plane (Profit > 0%) calculated and drawn.")
    else:
        print("Could not draw separation plane: Data only contains one class (all Profitable or all Loss).")

    ax.set_xlabel('History X (x1: Volatility Ratio)')
    ax.set_ylabel('History Y (y1: Price Delta Threshold)')
    ax.set_zlabel('History Z (z1: Window Span)')
    ax.set_title('3D Parameter Space: Historical Anomaly vs. Strategy Profit\n(Blue Plane separates >0% Profit)')
    
    plt.tight_layout()
    plt.savefig('profit_3d_graph.pdf', format='pdf', bbox_inches='tight')    
    print("3D Plot saved to 'profit_3d_graph.pdf'.")
    plt.show()

def run_global_parameter_search(data_bnl):
    """
    Main loop to run parameter search strictly in memory using the combined functions.
    """
    print("Starting in-memory global 3D parameter search...")
    
    x1_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    y1_list = [0.01, 0.02, 0.03, 0.04, 0.05]
    z1_list = [1, 2, 3, 4, 5]
    
    # x_f_list = [0.3, 0.4, 0.5]
    # y_f_list = [0.03, 0.04, 0.05]
    # z_f_list = [2, 3, 4]


    x_f_list = [0.4]
    y_f_list = [0.05]
    z_f_list = [3]
    summary_data = []
    
    total_outer = len(x1_list) * len(y1_list) * len(z1_list)
    print(f"Testing {total_outer} historical parameter combinations in memory...")
    
    for x1, y1, z1 in itertools.product(x1_list, y1_list, z1_list):
        inner_results = []
        
        for x_f, y_f, z_f in itertools.product(x_f_list, y_f_list, z_f_list):
            
            x_thresh = (x1, x_f)
            y_thresh = (y1, y_f)
            z_window = (z1, z_f)
            
            metrics = evaluate_investment_in_memory(data_bnl, x_thresh, y_thresh, z_window)
            
            # Filter out extreme noise or zero-signal runs based on total timeframe
            inner_results.append(metrics)
                
        if inner_results:
            df_inner = pd.DataFrame(inner_results)
            
            # Average the actual PROFIT rather than the win rate
            avg_profit = df_inner["Strategy_Profit(%)"].mean()
            avg_signals = df_inner["Strategy_Pos_label_Profit(%)"].mean()
            
            summary_data.append({
                "x1": x1,
                "y1": y1,
                "z1": z1,
                "Avg_Profit(%)": avg_profit,
                "Strategy_Pos_label_Profit(%)": avg_signals
            })
            
    df_summary = pd.DataFrame(summary_data)
    
    if df_summary.empty:
        print("No valid configurations found.")
        return df_summary
        
    print("\nGlobal Search Complete. Top 5 Configurations (by Max Profit):")
    df_summary = df_summary.sort_values(by="Avg_Profit(%)", ascending=False).reset_index(drop=True)
    print(df_summary.head(5).to_string())
    
    df_summary.to_csv("anomaly_profit_search.csv", index=False)
    print("\nFull results saved to 'anomaly_profit_search.csv'")
    
    return df_summary

# def run_parameter_search(features):
#     print("Starting investment-focused parameter search...")
    
#     # Define the parameter grid
#     # x_pairs = [(0.2, 0.1), (0.3, 0.15), (0.4, 0.2), (0.4, 0.1), (0.15, 0.4)]
#     # y_pairs = [(0.03, 0.03), (0.05, 0.04), (0.08, 0.07), (0.08, 0.04), (0.03, 0.07)]
#     # z_pairs = [(5, 1), (3, 1), (5, 3), (3, 3), (3, 5)]
#     # x_pairs = [(0.1, 0.3), (0.1, 0.4), (0.1, 0.5), (0.1, 0.6), (0.1, 0.7)]
#     # y_pairs = [(0.01, 0.03), (0.01, 0.04), (0.01, 0.05), (0.01, 0.06), (0.01, 0.07)]
#     # z_pairs = [(1, 1), (1, 3), (1, 4), (1, 5)]
#     x1=0.3
#     y1=0.05
#     z1=2
#     x_pairs = [(x1, 0.3), (x1, 0.4), (x1, 0.5), (x1, 0.6), (x1, 0.7)]
#     y_pairs = [(y1, 0.03), (y1, 0.04), (y1, 0.05), (y1, 0.06), (y1, 0.07)]
#     z_pairs = [(z1, 1), (z1, 3), (z1, 4), (z1, 5)]
#     results = []
    
#     total_combinations = len(x_pairs) * len(y_pairs) * len(z_pairs)
#     print(f"Testing {total_combinations} combinations...")
    
#     for x, y, z in itertools.product(x_pairs, y_pairs, z_pairs):
#         metrics = evaluate_investment_profitability(features, x, y, z)
        
#         # Filter: We only want parameters that actually generate signals, 
#         # but don't signal all the time (e.g., > 10 signals, < 5000 signals)
#         if 10 < metrics["Total_Signals"] < 5000:
#             results.append(metrics)
            
#     df_results = pd.DataFrame(results)
    
#     if df_results.empty:
#         print("No parameters generated valid signals within the thresholds. Try adjusting bounds.")
#         return df_results
        
#     # Sort by the most profitable Long strategy (Win Rate)
#     # Secondary sort by Avoid Rate (How well it skips losing trades)
#     df_results = df_results.sort_values(
#         by=["Win_Rate(%)", "Avoid_Rate(%)"], 
#         ascending=[False, False]
#     ).reset_index(drop=True)
    
#     print("\nSearch Complete. Top 10 'Most Profitable' Configurations:")
#     print(df_results.head(10).to_string())
    
#     df_results.to_csv("anomaly_profit_search.csv", index=False)
#     print("\nFull results saved to 'anomaly_profit_search.csv'")
    
#     return df_results

# --- EXAMPLE USAGE ---
# Just pass the same variables you used for labeling
features, stock_list = create_stock_data_numpy(save_path)
run_global_parameter_search(features)

# csv_file_path = "analysis/global_summary_profit_search.csv"
# print(f"Loading data from {csv_file_path}...")
# df_loaded_summary = pd.read_csv(csv_file_path)
# draw_3d_separation_graph(df_loaded_summary)