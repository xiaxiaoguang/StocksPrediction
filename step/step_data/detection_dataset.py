import os
import torch
import pickle
from torch.utils.data import Dataset

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

class AnomalyDetectionDataset(Dataset):
    """Time series anomaly detection dataset."""

    def __init__(self, data_file_path: str, index_file_path: str, label_file_path: str, mode: str) -> None:
        """
        Init the dataset for anomaly detection.
        """
        super().__init__()
        assert mode in ["train", "valid", "test"], "error mode"
        self._check_if_file_exists(data_file_path, index_file_path, label_file_path)
        
        # Read raw data (normalized features)
        data = load_pkl(data_file_path)
        self.data = torch.from_numpy(data["processed_data"]).float()

        # Read binary labels
        label_data = load_pkl(label_file_path)["processed_data"]
        self.label = torch.from_numpy(label_data).float() 
        
        # Read index
        self.index = load_pkl(index_file_path)[mode]
        
        # The length of the sequence is dynamically inferred from the first index tuple
        first_idx = self.index[0]
        self.seq_len = first_idx[1] - first_idx[0]

    def _check_if_file_exists(self, data_file_path: str, index_file_path: str, label_file_path: str):
        if not os.path.isfile(data_file_path):
            raise FileNotFoundError(f"Cannot find data file {data_file_path}")
        if not os.path.isfile(index_file_path):
            raise FileNotFoundError(f"Cannot find index file {index_file_path}")
        if not os.path.isfile(label_file_path):
            raise FileNotFoundError(f"Cannot find label file {label_file_path}")
        
    def __getitem__(self, index: int) -> tuple:
        """Get a sample.
        Returns:
            tuple: (history_data, label)
            - history_data shape: (seq_len, N, C)
            - label shape: (1,) -> Binary 0 or 1
        """
        # idx = (start_idx, pivot_idx)
        idx = list(self.index[index])
        start_idx, pivot_idx = idx[0], idx[1]
        
        # History fed to the model (Shape: Seq_Len x N x 1)
        history_data = self.data[start_idx:pivot_idx]
        
        # Target Label at the pivot point (Shape: 1)
        label = self.label[pivot_idx]

        return history_data, label

    def __len__(self):
        return len(self.index)