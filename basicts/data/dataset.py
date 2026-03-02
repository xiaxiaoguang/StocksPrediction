import os
import sys
import torch
from torch.utils.data import Dataset


from ..utils import load_pkl
sys.path.append('/home/srwang/trend_forecasting')
from dataprovider.data_factory import data_provider
from .dataArgs import load_arg

class NewTSDatasets(Dataset):
    def __init__(self, data_file_path: str, index_file_path: str, label_file_path : str, mode) -> None:
        super().__init__()
        assert mode in ["train", "valid", "test"], "error mode"
        data_file_path = data_file_path.split('/')[1]
        data_args = load_arg(data_file_path)
        
        if mode == 'train':
            self.dataset, self.dataloader = data_provider(data_args, 'train')
        if mode == 'valid':
            self.dataset, self.dataloader = data_provider(data_args, 'val')
        if mode == 'test':
            self.dataset, self.dataloader = data_provider(data_args, 'test')

    def __getitem__(self, index: int) -> tuple:
        return self.dataset.__getitem__(index)
    def __len__(self):
        return len(self.dataset)

class TimeSeriesForecastingDataset(Dataset):
    """Time series forecasting dataset."""

    def __init__(self, data_file_path: str, index_file_path: str, label_file_path : str, mode: str) -> None:
        super().__init__()
        assert mode in ["train", "valid", "test"], "error mode"
        self._check_if_file_exists(data_file_path, index_file_path,label_file_path)
        # read raw data (normalized)
        
        data = load_pkl(data_file_path)
        processed_data = data["processed_data"] 
        # breakpoint()
        if label_file_path is not None:
            label = load_pkl(label_file_path)["processed_data"]
            try :
                self.label = torch.from_numpy(label).float()
            except: 
                print("label is not numpy")
        else :
            self.label=None

        self.data = torch.from_numpy(processed_data).float()
        self.index = load_pkl(index_file_path)[mode]

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
        
        if (label_file_path is not None) and (not os.path.isfile(label_file_path)):
            raise FileNotFoundError("BasicTS can not find label file {0}".format(label_file_path))

    def __getitem__(self, index: int) -> tuple:
        """Get a sample.

        Args:
            index (int): the iteration index (not the self.index)

        Returns:
            tuple: (future_data, history_data), where the shape of each is L x N x C.
        """
        idx = list(self.index[index])
        if isinstance(idx[0], int):
            # continuous index
            history_data = self.data[idx[0]:idx[1]]
            future_data = self.data[idx[1]:idx[2]]
            if self.label is not None:
                label = self.label[idx[0]:idx[1]]
        # else:
        #     # discontinuous index or custom index
        #     # NOTE: current time $t$ should not included in the index[0]
        #     history_index = idx[0]    # list
        #     assert idx[1] not in history_index, "current time t should not included in the idx[0]"
        #     history_index.append(idx[1])
        #     history_data = self.data[history_index]
        #     future_data = self.data[idx[1], idx[2]]
        #     label = self.label[idx[1]]
        
        if self.label is not None:
            return future_data, history_data ,label
        else :
            return future_data, history_data

    def __len__(self):
        """Dataset length

        Returns:
            int: dataset length
        """

        return len(self.index)
