import os

import torch
from torch.utils.data import Dataset
from basicts.utils import load_pkl


class ForecastingDataset(Dataset):
    """Time series forecasting dataset."""

    def __init__(self, data_file_path: str, index_file_path: str,label_file_path: str ,mode: str, seq_len:int) -> None:
        """Init the dataset in the forecasting stage.

        Args:
            data_file_path (str): data file path.
            index_file_path (str): index file path.
            mode (str): train, valid, or test.
            seq_len (int): the length of long term historical data.
        """

        super().__init__()
        assert mode in ["train", "valid", "test"], "error mode"
        self._check_if_file_exists(data_file_path, index_file_path,label_file_path)
        # read raw data (normalized)
        data = load_pkl(data_file_path)
        processed_data = data["processed_data"]
        self.data = torch.from_numpy(processed_data).float()

        if label_file_path is not None :
            label = load_pkl(label_file_path)["processed_data"]
            self.label = torch.from_numpy(label).float() # exception 
        else :
            self.label = torch.ones_like(self.data)
        # read index
        self.index = load_pkl(index_file_path)[mode]
        for idx,a in enumerate(self.index):
            if a[0]>=seq_len:
                self.index = self.index[idx:]
                break # length of long term historical data
        self.seq_len = seq_len# mask
        self.mask = torch.zeros(self.seq_len, self.data.shape[1], self.data.shape[2])

    def _check_if_file_exists(self, data_file_path: str, index_file_path: str,label_file_path):
        """Check if data file and index file exist.

        Args:
            data_file_path (str): data file path
            index_file_path (str): index file path

        Raises:
            FileNotFoundError: no data file
            FileNotFoundError: no index file
        """

        if not os.path.isfile(data_file_path):
            raise FileNotFoundError("BasicTS can not find data file {0}".format(data_file_path))
        if not os.path.isfile(index_file_path):
            raise FileNotFoundError("BasicTS can not find index file {0}".format(index_file_path))
        if label_file_path is not None and not os.path.isfile(label_file_path):
            raise FileNotFoundError("BasicTS can not find label file {0}".format(label_file_path))
        
    def __getitem__(self, index: int) -> tuple:
        """Get a sample.

        Args:
            index (int): the iteration index (not the self.index)

        Returns:
            tuple: (future_data, history_data), where the shape of each is L x N x C.
        """

        idx = list(self.index[index])
        history_data = self.data[idx[0]:idx[1]]
        future_data = self.data[idx[1]:idx[2]]
        label = self.label[idx[1]:idx[2]]

        if idx[1] - self.seq_len < 0:
            NotImplementedError
        else:
            long_history_data = self.data[idx[1] - self.seq_len:idx[1]]     

        return future_data, history_data, long_history_data, label

    def __len__(self):
        """Dataset length

        Returns:
            int: dataset length
        """

        return len(self.index)
