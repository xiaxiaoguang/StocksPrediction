import math
import torch
import numpy as np
import functools
from typing import Union, Tuple, Optional

from metric import *
from basicts.runners.base_runner import BaseRunner

class AnomalyDetectionRunner(BaseRunner):
    """
    Runner for binary anomaly detection in time series.
    Features:
        - Predicts 0 (Normal) or 1 (Anomaly) for a given future window.
        - Metrics: Accuracy, Precision, Recall, F1-Score, AUC.
        - Best model is selected based on the highest F1-Score at validation.
        - Handles severe class imbalance inherent in anomaly detection.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.z_f = cfg["DATAPARAM"]['z_f']
        self.dataset_name = cfg["DATASET_NAME"]
        self.dataset_type = cfg["DATASET_TYPE"]
        self.evaluate_on_gpu = cfg["TEST"].get("USE_GPU", True)
        self.use_local=False
        # Loss Function (e.g., nn.BCEWithLogitsLoss)
        self.loss = cfg["TRAIN"]["LOSS"]
        
        # We no longer need MAE/RMSE. We register classification metrics.
        self.metrics = {
            "F1": detection_f1,
            "Accuracy": detection_accuracy,
            "Precision": detection_precision,
            "Recall": detection_recall,
            "AUC": detection_auc,
        }

    def init_training(self, cfg: dict):
        super().init_training(cfg)
        for key in self.metrics.keys():
            self.register_epoch_meter("train_" + key, "train", "{:.4f}")
        if cfg.get("TEST_ONLY",False):
            self.register_epoch_meter("test_strategy_return","{:.4f}")
            self.register_epoch_meter("test_sharpe","{:.4f}")
            self.test_process(cfg)
            quit("TEST FINISHED!")

    def init_validation(self, cfg: dict):
        super().init_validation(cfg)
        for key in self.metrics.keys():
            self.register_epoch_meter("val_" + key, "val", "{:.4f}")

    def init_test(self, cfg: dict):
        super().init_test(cfg)
        for key in self.metrics.keys():
            self.register_epoch_meter("test_" + key, "test", "{:.4f}")

        self.register_epoch_meter("test_strategy_return","{:.4f}")
        self.register_epoch_meter("test_sharpe","{:.4f}")
    
    @staticmethod
    def _get_suffix(cfg: dict):
        """Helper to extract parameters and build the filename suffix."""
        p = cfg["DATAPARAM"]
        return f"_{p['seq_len']}_{p['num_anomalies']}_his{p['x_h']}_{p['y_h']}_{p['z_h']}_fut{p['x_f']}_{p['y_f']}_{p['z_f']}_"

    def build_train_dataset(self, cfg: dict):
        suffix = self._get_suffix(cfg)
        data_dir = cfg['TRAIN']['DATA']['DIR']
        
        dataset_args = cfg.get("DATASET_ARGS", {})
        dataset_args["data_file_path"] = f"{data_dir}/data_anomaly{suffix}.pkl"
        dataset_args["index_file_path"] = f"{data_dir}/index_anomaly{suffix}.pkl"
        dataset_args["label_file_path"] = f"{data_dir}/label_anomaly{suffix}.pkl"
        dataset_args["scaler_file_path"] = f"{data_dir}/scaler_anomaly{suffix}.pkl"
        dataset_args["z_f"] = cfg["DATAPARAM"]['z_f']

        dataset_args["mode"] = "train"
        dataset = cfg["DATASET_CLS"](**dataset_args)
        self.iter_per_epoch = math.ceil(len(dataset) / cfg["TRAIN"]["DATA"]["BATCH_SIZE"])
        return dataset

    @staticmethod
    def build_val_dataset(cfg: dict):
        suffix = AnomalyDetectionRunner._get_suffix(cfg)
        data_dir = cfg['VAL']['DATA']['DIR']
        
        dataset_args = cfg.get("DATASET_ARGS", {})
        dataset_args["data_file_path"] = f"{data_dir}/data_anomaly{suffix}.pkl"
        dataset_args["index_file_path"] = f"{data_dir}/index_anomaly{suffix}.pkl"
        dataset_args["label_file_path"] = f"{data_dir}/label_anomaly{suffix}.pkl"
        dataset_args["scaler_file_path"] = f"{data_dir}/scaler_anomaly{suffix}.pkl"   
        dataset_args["z_f"] = cfg["DATAPARAM"]['z_f']

        dataset_args["mode"] = "valid"
        return cfg["DATASET_CLS"](**dataset_args)

    @staticmethod
    def build_test_dataset(cfg: dict):
        suffix = AnomalyDetectionRunner._get_suffix(cfg)
        data_dir = cfg['TEST']['DATA']['DIR']
        
        dataset_args = cfg.get("DATASET_ARGS", {})
        dataset_args["data_file_path"] = f"{data_dir}/data_anomaly{suffix}.pkl"
        dataset_args["index_file_path"] = f"{data_dir}/index_anomaly{suffix}.pkl"
        dataset_args["label_file_path"] = f"{data_dir}/label_anomaly{suffix}.pkl"
        dataset_args["scaler_file_path"] = f"{data_dir}/scaler_anomaly{suffix}.pkl"
        dataset_args["z_f"] = cfg["DATAPARAM"]['z_f']

        dataset_args["mode"] = "test"
        return cfg["DATASET_CLS"](**dataset_args)


    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        """
        Must return (logits, labels)
        Logits shape: [B, 1]
        Labels shape: [B, 1]
        """
        raise NotImplementedError()

    def calculate_metrics(self, logits, labels: torch.Tensor, prefix: str):
        """Helper to compute and update classification metrics purely on the GPU."""
        # 1. Calculate probabilities and predictions on the GPU
        probs = {}
        preds = {}
        for i in ['local','global']:
            probs[i] = torch.sigmoid(logits[i])
            preds[i] = (probs[i] > 0.5).float()

        # 2. Iterate through metrics without leaving the device
        for metric_name, metric_func in self.metrics.items():
            if metric_name == "AUC":
                # AUC needs the continuous probabilities
                val = metric_func(probs, labels)
            else:
                # Precision, Recall, F1, Acc need the hard predictions
                val = metric_func(preds, labels)
            
            # Only pull the final scalar value to the CPU for logging
            if val is not None:
                self.update_epoch_meter(f"{prefix}_{metric_name}", val.item())

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        logits, labels = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)
        
        # Calculate Loss (NO INVERSE SCALING)
        loss = self.loss(logits, labels)
        
        # Calculate Metrics
        self.calculate_metrics(logits, labels, prefix="train")
        return loss

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        logits, labels = self.forward(data=data, epoch=None, iter_num=iter_index, train=False)
        self.calculate_metrics(logits, labels, prefix="val")
        
    # @master_only 
    def test(self, use_local: bool = True):
        mode_str = "Localized" if use_local else "Global"
        self.logger.info(f"Starting {mode_str} financial evaluation on test dataset...")
        
        all_logits = {'global': []}
        all_labels = {'global': []}
        if use_local:
            all_logits['local'] = []
            all_labels['local'] = []
            
        all_returns = []
        # 1. Collect Predictions and Actual Returns
        for _, data in enumerate(self.test_data_loader):
            inputs, labels, forward_returns = data 
            logits, _ = self.forward((inputs, labels), epoch=None, iter_num=None, train=False)
            if isinstance(logits, dict):
                all_logits['global'].append(logits['global'])
                if use_local and 'local' in logits:
                    all_logits['local'].append(logits['local'])
            else:
                all_logits['global'].append(logits)
                
            if isinstance(labels, dict):
                all_labels['global'].append(labels['global'])
                if use_local and 'local' in labels:
                    all_labels['local'].append(labels['local'])
            else:
                all_labels['global'].append(labels)
                
            all_returns.append(forward_returns)
        # Concatenate everything
        all_logits['global'] = torch.cat(all_logits['global'], dim=0)
        all_labels['global'] = torch.cat(all_labels['global'], dim=0)
        if use_local:
            all_logits['local'] = torch.cat(all_logits['local'], dim=0)
            all_labels['local'] = torch.cat(all_labels['local'], dim=0)
            
        # Shape is strictly [Total_B, N] based on your dataloader
        all_returns = torch.cat(all_returns, dim=0).cpu().numpy() 

        # 2. Standard ML Metrics
        self.logger.info("--- Standard ML Metrics ---")
        if use_local:
            for metric_name, metric_func in self.metrics.items():
                metric_val = metric_func(all_logits, all_labels).cpu().numpy()
                self.update_epoch_meter("test_" + metric_name, float(metric_val))
                self.logger.info(f"Test {metric_name}: {metric_val:.4f}")
        else:
            probs = torch.sigmoid(all_logits['global']).cpu()
            preds = (probs > 0.5).float()
            
            targets = all_labels['global'].float().cpu()
            for metric_name, metric_func in self.metrics.items():
                if metric_name == "AUC":
                    metric_val = metric_func(probs, targets)
                else:
                    metric_val = metric_func(preds, targets)
                    metric_val = metric_val.cpu().numpy()
                self.update_epoch_meter("test_" + metric_name, float(metric_val))
                self.logger.info(f"Test {metric_name}: {metric_val:.4f}")

        # 3. Real-World Financial Backtest Simulation
        self.logger.info(f"--- {mode_str} Directional Oracle Simulation (55% Win Rate) ---")
        
        Total_B, N = all_returns.shape
        z_f = self.z_f 
        transactions = (1 - 1e-4) # 1e-4 transaction fee
        
        # The mathematical edge of a 55% win rate machine: 0.55 - 0.45 = 0.10
        oracle_edge = 0.10 
        # Initialize Trackers: Baseline (trade everything) vs Model (trade only anomalies)
        base_cash, model_cash = 1.0, 1.0
        base_active_trades, model_active_trades = [], []
        
        base_wealth_history, model_wealth_history = [], []
        base_returns_recorded, model_returns_recorded = [], []
        
        base_active_days, model_active_days = 0, 0
        
        # RESTORED: Track average stocks held
        total_base_stocks_held = 0
        total_model_stocks_held = 0
        # Pre-compute Model Anomaly Selections
        if use_local:
            local_probs = torch.sigmoid(all_logits['local']).cpu().numpy()
            model_positions = (local_probs > 0.5).astype(float) 
        else:
            global_probs = torch.sigmoid(all_logits['global']).cpu().numpy().flatten()
            model_positions = (global_probs > 0.5).astype(float)
        kill = 0
        
        for t in range(Total_B):
            # --- A. Process Unlocks ---
            still_active_base = []
            for trade in base_active_trades:
                if trade['unlock_time'] <= t: base_cash += trade['expected_payoff']
                else: still_active_base.append(trade)
            base_active_trades = still_active_base
            
            still_active_model = []
            for trade in model_active_trades:
                if trade['unlock_time'] <= t: model_cash += trade['expected_payoff']
                else: still_active_model.append(trade)
            model_active_trades = still_active_model

            # --- B. Calculate Current Wealth ---
            base_current_wealth = base_cash + sum(tr['basis'] for tr in base_active_trades)
            model_current_wealth = model_cash + sum(tr['basis'] for tr in model_active_trades)
            
            base_wealth_history.append(base_current_wealth)
            model_wealth_history.append(model_current_wealth)

            # --- C. Calculate Expected Oracle Returns for Day/Minute t ---
            t_base_return, t_model_return = 0.0, 0.0
            base_signal, model_signal = False, False
            
            active_mask = (all_returns[t] != 0.0)
            
            if use_local:
                # 1. BASELINE: Trade all active stocks
                active_count = np.sum(active_mask)
                if active_count > 0:
                    base_signal = True
                    base_active_days += 1
                    total_base_stocks_held += active_count
                    # Exact average magnitude of ALL active stocks
                    all_active_returns = all_returns[t][active_mask]
                    t_base_return = np.mean(oracle_edge * np.abs(all_active_returns))

                # 2. MODEL: Trade ONLY selected anomalies
                selected_mask = (model_positions[t] > 0) & active_mask
                model_active_count = np.sum(selected_mask)
                if model_active_count > 0:
                    model_signal = True
                    model_active_days += 1
                    total_model_stocks_held += model_active_count
                    # CORRECTED: Isolate the exact returns of selected stocks and average them directly
                    selected_returns = all_returns[t][selected_mask]
                    if np.mean(np.abs(selected_returns)) > 0.03 and kill < 10:
                        self.logger.info(f"in minute {t} we trade {np.where(selected_mask)}")
                        kill += 1
                    t_model_return = np.mean(oracle_edge * np.abs(selected_returns))
            else:
                # Global Logic (Trading the Market Index)
                active_count = np.sum(active_mask)
                if active_count > 0:
                    base_signal = True
                    base_active_days += 1
                    # Magnitude of the global market return
                    market_return = np.mean(all_returns[t][active_mask])
                    t_base_return = oracle_edge * np.abs(market_return)
                    
                if model_positions[t] > 0 and active_count > 0:
                    model_signal = True
                    model_active_days += 1
                    market_return = np.mean(all_returns[t][active_mask])
                    t_model_return = oracle_edge * np.abs(market_return)
            # --- D. Execute Trades ---
            unlock_t = t + z_f 
            
            # Baseline machine invests 1/z_f on everything
            if base_signal:
                base_returns_recorded.append(t_base_return)
                base_invest_amt = min(base_cash, base_current_wealth / z_f)
                if base_invest_amt > 0:
                    base_cash -= base_invest_amt
                    base_active_trades.append({
                        'unlock_time': unlock_t, 
                        'basis': base_invest_amt * transactions, 
                        'expected_payoff': base_invest_amt * (1 + t_base_return) * transactions
                    })
                
            # Model-Filtered machine invests 1/z_f ONLY on anomalies
            if model_signal:
                model_returns_recorded.append(t_model_return)
                model_invest_amt = min(model_cash, model_current_wealth / z_f)
                if model_invest_amt > 0:
                    model_cash -= model_invest_amt
                    model_active_trades.append({
                        'unlock_time': unlock_t, 
                        'basis': model_invest_amt * transactions, 
                        'expected_payoff': model_invest_amt * (1 + t_model_return) * transactions
                    })
            
        # --- E. Post-Loop Cleanup ---
        final_base_wealth = base_cash + sum(tr['expected_payoff'] for tr in base_active_trades)
        final_model_wealth = model_cash + sum(tr['expected_payoff'] for tr in model_active_trades)
        
        base_wealth_history.append(final_base_wealth)
        model_wealth_history.append(final_model_wealth)
        # 4. Calculate Shared Metrics
        def calc_metrics(wealth_history, returns_recorded, active_days):
            wealth_arr = np.array(wealth_history)
            cum_return = wealth_arr[-1] - 1.0
            
            daily_returns = np.diff(wealth_arr) / wealth_arr[:-1]
            mean_ret = np.mean(daily_returns) if len(daily_returns) > 0 else 0
            std_ret = np.std(daily_returns) + 1e-8 if len(daily_returns) > 0 else 1e-8
            sharpe = (mean_ret / std_ret) * np.sqrt(252) # Adjust 252 for minute-data if needed
            
            run_max = np.maximum.accumulate(wealth_arr)
            drawdowns = (wealth_arr - run_max) / run_max
            max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0
            
            exposure = active_days / Total_B if Total_B > 0 else 0
            return cum_return, sharpe, max_dd, exposure
        base_cum, base_sharpe, base_dd, base_exp = calc_metrics(base_wealth_history, base_returns_recorded, base_active_days)
        mod_cum, mod_sharpe, mod_dd, mod_exp = calc_metrics(model_wealth_history, model_returns_recorded, model_active_days)

        # Log Side-by-Side Metrics
        self.logger.info(f"Simulation Holding Period (z_f): {z_f} steps")
        self.logger.info("-----------------------------------------------------------------")
        self.logger.info("METRIC                 | 55% ORACLE (ALL MIN) | 55% ORACLE (ANOMALY)")
        self.logger.info(f"Cumulative Return      | {base_cum * 100:>15.2f}% | {mod_cum * 100:>16.2f}%")
        self.logger.info(f"Annualized Sharpe      | {base_sharpe:>16.2f} | {mod_sharpe:>17.2f}")
        self.logger.info(f"Maximum Drawdown       | {base_dd * 100:>15.2f}% | {mod_dd * 100:>16.2f}%")
        self.logger.info(f"Time in Market (Exp.)  | {base_exp * 100:>15.2f}% | {mod_exp * 100:>16.2f}%")
        
        # RESTORED: Logging the average stocks held safely
        if use_local:
            avg_base_stocks = total_base_stocks_held / max(1, base_active_days)
            avg_mod_stocks = total_model_stocks_held / max(1, model_active_days)
            self.logger.info(f"Avg Stocks Held/Day    | {avg_base_stocks:>16.1f} | {avg_mod_stocks:>17.1f}")
            
        self.logger.info("-----------------------------------------------------------------")
            
        # Save metrics
        self.update_epoch_meter("test_strategy_return", float(mod_cum))
        self.update_epoch_meter("test_sharpe", float(mod_sharpe))

    # @master_only
    def on_validating_end(self, train_epoch: Optional[int]):
        if train_epoch is not None:
            self.save_best_model(train_epoch, "val_F1", greater_best=True)