import os
import torch
import pickle
from torch.utils.data import Dataset

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

class AnomalyDetectionDataset(Dataset):
    """Time series anomaly detection dataset with financial return extraction."""

    def __init__(self, data_file_path: str, index_file_path: str, label_file_path: str, scaler_file_path: str, mode: str, z_f: int = 5) -> None:
        """
        Init the dataset for anomaly detection.
        Args:
            z_f: The future window size used to calculate the forward return.
        """
        super().__init__()
        assert mode in ["train", "valid", "test"], "error mode"
        self._check_if_file_exists(data_file_path, index_file_path, label_file_path, scaler_file_path)
        self.mode = mode
        # 1. Read raw data (normalized features)
        data = load_pkl(data_file_path)
        self.data = torch.from_numpy(data["processed_data"]).float()
        # 2. Read binary labels
        label_data = load_pkl(label_file_path)["processed_data"]
        # self.label = torch.from_numpy(label_data).float() 

        self.label = {
            'global': torch.from_numpy(label_data['global']).float(),
            'local': torch.from_numpy(label_data['local']).float()
        }        # 3. Read index
        
        self.local_ratio = label_data['local'].sum() / (label_data['local'].shape[0] * label_data['local'].shape[1])
        self.global_ratio = label_data['global'].sum() / (label_data['global'].shape[0] * label_data['global'].shape[1])
        self.pos_global = (1 - self.global_ratio) / self.global_ratio
        self.pos_local  = (1 - self.local_ratio) / self.local_ratio
        print('local and global ratio : ',self.local_ratio,self.global_ratio,(1-self.local_ratio)/self.local_ratio,(1-self.global_ratio)/self.global_ratio)

        self.index = load_pkl(index_file_path)[mode]

        # The length of the sequence is dynamically inferred
        first_idx = self.index[0]
        self.seq_len = first_idx[1] - first_idx[0]
        
        # 4. NEW: Load Scaler Parameters for unscaling prices
        scaler_data = load_pkl(scaler_file_path)
        self.mean = float(scaler_data['args']['mean'])
        self.std = float(scaler_data['args']['std'])
        self.z_f = z_f # Future window size

    def _check_if_file_exists(self, data_file_path: str, index_file_path: str, label_file_path: str, scaler_file_path: str):
        if not os.path.isfile(data_file_path):
            raise FileNotFoundError(f"Cannot find data file {data_file_path}")
        if not os.path.isfile(index_file_path):
            raise FileNotFoundError(f"Cannot find index file {index_file_path}")
        if not os.path.isfile(label_file_path):
            raise FileNotFoundError(f"Cannot find label file {label_file_path}")
        if not os.path.isfile(scaler_file_path):
            raise FileNotFoundError(f"Cannot find scaler file {scaler_file_path}")
        
    def __getitem__(self, index: int) -> tuple:
        """Get a sample.
        Returns:
            tuple: (history_data, label, market_forward_return)
            - history_data shape: (seq_len, N, C)
            - label shape: (1,) -> Binary 0 or 1
            - market_forward_return shape: (N,) -> True percentage return
        """
        idx = list(self.index[index])
        start_idx, pivot_idx = idx[0], idx[1]
        
        # History fed to the model (Shape: Seq_Len x N x 1)
        history_data = self.data[start_idx:pivot_idx]
        # Target Label at the pivot point (Shape: 1)
        # label = self.label[pivot_idx]
        label = {
            'global': self.label['global'][pivot_idx],
            'local': self.label['local'][pivot_idx]
        }

        # --- FINANCIAL BACKTEST DATA EXTRACTION ---
        # 1. Get future index (clamp to max length just in case)
        if self.mode == 'test':
            future_idx = min(pivot_idx + self.z_f, len(self.data) - 1)
            # Ensure shape is (N,) and data is tensor
            scaled_p_t = self.data[pivot_idx].squeeze(-1).clone().detach() 
            scaled_p_future = self.data[future_idx].squeeze(-1).clone().detach()
            if not isinstance(self.mean, torch.Tensor):
                mean_tensor = torch.tensor(self.mean, device=scaled_p_t.device, dtype=torch.float32)
                std_tensor = torch.tensor(self.std, device=scaled_p_t.device, dtype=torch.float32)
            else:
                mean_tensor = self.mean.to(scaled_p_t.device)
                std_tensor = self.std.to(scaled_p_t.device)
                
            # 2. Unscale to get the RAW actual prices
            p_t = (scaled_p_t * std_tensor) + mean_tensor
            p_future = (scaled_p_future * std_tensor) + mean_tensor
            active_mask = torch.abs(p_t) > 1e-6
            safe_p_t = torch.where(active_mask, p_t, torch.ones_like(p_t))
            stock_returns = torch.where(active_mask, (p_future - p_t) / safe_p_t, torch.zeros_like(p_t))
            return history_data, label, stock_returns
        else :
            return history_data, label

    def __len__(self):
        return len(self.index)