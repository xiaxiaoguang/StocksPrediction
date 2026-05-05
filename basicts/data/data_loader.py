import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import time
import warnings
import torch

warnings.filterwarnings('ignore')

class Dataset_Finance(Dataset):
    def __init__(self, root_path='./', flag='train', size=None, data_name='index',
                 data_path='index_data.pkl', stock_num=41,
                 scale=True, label_num=3):
        # size [seq_len, label_len, pred_len]
        # info
        if size != None:
            self.seq_len = size[0]
            self.pred_len = size[1]
        else:
            self.seq_len = 60
            self.pred_len = 1
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.label_num = label_num

        self.train_from = pd.to_datetime('2015-01-01')
        self.val_from = pd.to_datetime('2022-06-01')
        self.test_from = pd.to_datetime('2023-01-01')
        self.test_end = pd.to_datetime('2025-07-01')

        self.scale = scale
        self.scaler = None
        self.stock_num = stock_num

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        ohlcv_data = pd.read_pickle(os.path.join(self.root_path, self.data_path))
        # ohlcv_data['gt1'] = self.delay(ohlcv_data.Open, -2) / self.delay(ohlcv_data.Open, -1) - 1
        # ohlcv_data['gt3'] = self.delay(ohlcv_data.open, -3) / self.delay(ohlcv_data.open, -1) - 1 # 两天 0.02
        ohlcv_data['gt3'] = self.delay(ohlcv_data.open, -2) / self.delay(ohlcv_data.open, -1) - 1 # 一天 0.012
        # ohlcv_data['gt3'] = self.delay(ohlcv_data.open, -4) / self.delay(ohlcv_data.open, -1) - 1 # 三天 0.03
        ohlcv_data.vol = ohlcv_data.vol.astype(float)

        data = pd.DataFrame()
        data['rel_close'] = ohlcv_data.close / self.delay(ohlcv_data.close, 1) - 1
        data['rel_open'] = ohlcv_data.open / ohlcv_data.close - 1
        data['rel_high'] = ohlcv_data.high / ohlcv_data.close - 1
        data['rel_low'] = ohlcv_data.low / ohlcv_data.close - 1
        data['rel_pre_close'] = ohlcv_data.pre_close / ohlcv_data.close - 1
        data['rel_change'] = ohlcv_data.change / ohlcv_data.close - 1
        data['volume'] = ohlcv_data.vol / (self.ts_mean(ohlcv_data.vol, 42) + 1e-8) - 1
        data['amount'] = ohlcv_data.amount / (self.ts_mean(ohlcv_data.amount, 42) + 1e-8) - 1
        data['gt'] = ohlcv_data.gt3
        if self.label_num == 3:
            data['label'] = np.where(ohlcv_data.gt3 > 0.03, 2, np.where(ohlcv_data.gt3 < -0.03, 0, 1))
        else:
            # data['label'] = np.where((ohlcv_data.gt3 < 0.03) & (ohlcv_data.gt3 > -0.03), 0, 1)
            data['label'] = np.where(ohlcv_data.gt3 > -0.02, 0, 1)
        
        data = data.replace([np.inf, -np.inf], value=np.nan)
        data = data.reset_index().set_index(['trade_date', 'code']).sort_index()
    
        if self.scale:
            feature_cols = data.columns.drop(['gt', 'label'])
            data_dropna = data.loc[:self.val_from-pd.Timedelta(days=1), feature_cols].dropna()
            mean, std = data_dropna.mean(), data_dropna.std()
            data[feature_cols] = (data[feature_cols] - mean) / std
            data = self.replace_dtypes(data)
            self.mean = mean
            self.std = std

        data = data.fillna(0)
                
        #dataset split
        # if self.set_type == 0:
        #     self.data = data.loc[self.train_from:self.val_from-pd.Timedelta(days=1)]
        # elif self.set_type == 1:
        #     self.data = data.loc[self.val_from-pd.Timedelta(days=self.seq_len+self.pred_len-1)
        #                          :self.test_from-pd.Timedelta(days=1)]
        # else:
        #     self.data = data.loc[self.test_from-pd.Timedelta(days=self.seq_len+self.pred_len-1)
        #                          :self.test_end-pd.Timedelta(days=1)]

        # self.Ts = self.data.index.unique(0)
        total_Ts = data.index.unique(0)
        start_idx = 0
        end_idx = 0
        if self.set_type == 0:
            start_date = self.train_from
            end_date = self.val_from
        elif self.set_type == 1:
            start_date = self.val_from
            end_date = self.test_from
        else:
            start_date = self.test_from
            end_date = self.test_end
        start_idx = -1
        end_idx = -1
        for t in range(0,len(total_Ts)):
            if start_date < total_Ts[t]:
                start_idx = t
                break
        for t in range(0,len(total_Ts)):
            if end_date < total_Ts[t]:
                end_idx = t - 1
                break
        if start_idx < self.seq_len - 1:
            start_idx = self.seq_len - 1
        self.Ts = total_Ts[start_idx-self.seq_len+1:end_idx]
        self.data = data.loc[self.Ts[0]:self.Ts[-1]-pd.Timedelta(days=1)]

    def __getitem__(self, index):
        s_begin = self.Ts[index]
        s_end = self.Ts[index + self.seq_len - 1] # -1是因为以时间为索引[a,b]会取到b
        r_begin = self.Ts[index + self.seq_len - 1] # -1是因为当天的gt1使用后天和明天的收益计算的
        r_end = self.Ts[index + self.seq_len + self.pred_len - 2] # -2是上面两个-1累计效果
        x = self.data.drop(columns=['gt', 'label']).loc[s_begin:s_end].fillna(0.).values
        y = self.data[['gt','label']].loc[r_begin:r_end].fillna(0.).values
        x = x.reshape((self.seq_len, self.stock_num, -1)).transpose((1,0,2))
        y = y.reshape((self.pred_len, self.stock_num, -1)).transpose((1,0,2))
        return x, y

    def __len__(self):
        return len(self.Ts) - self.seq_len - self.pred_len + 1
    
    def delay(self, series, d, level=1):
        d = int(d)
        return series.groupby(level=level, group_keys=False).shift(d)
    
    def ts_mean(self, series, d):
        d = int(d)
        return series.groupby(level=1, group_keys=False).rolling(d).mean().droplevel(0).sort_index()
    
    def replace_dtypes(self, data, rules={'float64': 'float32', 'int32': 'int64'}):
        for k, v in rules.items():
            cols = data.dtypes[data.dtypes == k].index
            data = data.astype({t: v for t in cols}, copy=False)
        return data

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import time
import warnings
import torch

warnings.filterwarnings('ignore')

class Dataset_Finance_new(Dataset):
    def __init__(self, root_path='./datasets/', flag='train', size=None, data_name='index',
                 data_path='index_data_feat_202411.pkl', stock_num=41,
                 scale=True, label_num=3):
        # size [seq_len, label_len, pred_len]
        # info
        if size != None:
            self.seq_len = size[0]
            self.pred_len = size[1]
        else:
            self.seq_len = 60
            self.pred_len = 1
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 1}
        self.set_type = type_map[flag]
        self.label_num = label_num

        self.train_from = pd.to_datetime('2015-01-01')
        self.val_from = pd.to_datetime('2023-01-01')
        self.test_from = pd.to_datetime('2025-06-01')
        self.test_end = pd.to_datetime('2029-07-01')

        self.scale = scale
        self.scaler = None
        self.stock_num = stock_num

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        ohlcv_data = pd.read_pickle(os.path.join(self.root_path, self.data_path))
        # ohlcv_data['gt1'] = self.delay(ohlcv_data.Open, -2) / self.delay(ohlcv_data.Open, -1) - 1 
        ohlcv_data['gt3'] = self.delay(ohlcv_data.open, -3) / self.delay(ohlcv_data.open, -1) - 1 # 两天 0.02
        # ohlcv_data['gt3'] = self.delay(ohlcv_data.open, -2) / self.delay(ohlcv_data.open, -1) - 1 # 一天 0.012
        # ohlcv_data['gt3'] = self.delay(ohlcv_data.open, -4) / self.delay(ohlcv_data.open, -1) - 1 # 三天 0.03
        ohlcv_data.volume = ohlcv_data.volume.astype(float)

        data = ohlcv_data
        data['gt'] = ohlcv_data.gt3
        if self.label_num == 3:
            data['label'] = np.where(ohlcv_data.gt3 > 0.03, 2, np.where(ohlcv_data.gt3 < -0.03, 0, 1))
        else:
            # data['label'] = np.where((ohlcv_data.gt3 < 0.03) & (ohlcv_data.gt3 > -0.03), 0, 1)
            data['label'] = np.where(ohlcv_data.gt3 > -0.02, 0, 1)
        
        data = data.replace([np.inf, -np.inf], value=np.nan)
        data = data.reset_index().set_index(['trade_date', 'code']).sort_index()
    
        if self.scale:
            feature_cols = data.columns.drop(['gt', 'label'])
            data_dropna = data.loc[:self.val_from-pd.Timedelta(days=1), feature_cols].dropna()
            mean, std = data_dropna.mean(), data_dropna.std()
            data[feature_cols] = (data[feature_cols] - mean) / std
            data = self.replace_dtypes(data)
            self.mean = mean
            self.std = std

        data = data.fillna(0)
                
        #dataset split
        # if self.set_type == 0:
        #     self.data = data.loc[self.train_from:self.val_from-pd.Timedelta(days=1)]
        # elif self.set_type == 1:
        #     self.data = data.loc[self.val_from-pd.Timedelta(days=self.seq_len+self.pred_len-1)
        #                          :self.test_from-pd.Timedelta(days=1)]
        # else:
        #     self.data = data.loc[self.test_from-pd.Timedelta(days=self.seq_len+self.pred_len-1)
        #                          :self.test_end-pd.Timedelta(days=1)]

        # self.Ts = self.data.index.unique(0)
        total_Ts = data.index.unique(0)
        start_idx = 0
        end_idx = 0
        if self.set_type == 0:
            start_date = self.train_from
            end_date = self.val_from
        elif self.set_type == 1:
            start_date = self.val_from
            end_date = self.test_from
        else:
            start_date = self.test_from
            end_date = self.test_end
        start_idx = -1
        end_idx = -1
        for t in range(0,len(total_Ts)):
            if start_date < total_Ts[t]:
                start_idx = t
                break
        for t in range(0,len(total_Ts)):
            if end_date < total_Ts[t]:
                end_idx = t - 1
                break
        if start_idx < self.seq_len - 1:
            start_idx = self.seq_len - 1
        if end_idx == -1:
            self.Ts = total_Ts[start_idx-self.seq_len+1:]
        else:
            self.Ts = total_Ts[start_idx-self.seq_len+1:end_idx]
        self.data = data.loc[self.Ts[0]:self.Ts[-1]-pd.Timedelta(days=1)]

    def __getitem__(self, index):
        s_begin = self.Ts[index]
        s_end = self.Ts[index + self.seq_len - 1] # -1是因为以时间为索引[a,b]会取到b
        r_begin = self.Ts[index + self.seq_len - 1] # -1是因为当天的gt1使用后天和明天的收益计算的
        r_end = self.Ts[index + self.seq_len + self.pred_len - 2] # -2是上面两个-1累计效果
        x = self.data.drop(columns=['gt', 'label']).loc[s_begin:s_end].fillna(0.).values
        y = self.data[['gt','label']].loc[r_begin:r_end].fillna(0.).values
        x = x.reshape((self.seq_len, self.stock_num, -1)).transpose((1,0,2))
        y = y.reshape((self.pred_len, self.stock_num, -1)).transpose((1,0,2))
        return x, y

    def __len__(self):
        return len(self.Ts) - self.seq_len - self.pred_len + 1
    
    def delay(self, series, d, level=1):
        d = int(d)
        return series.groupby(level=level, group_keys=False).shift(d)
    
    def ts_mean(self, series, d):
        d = int(d)
        return series.groupby(level=1, group_keys=False).rolling(d).mean().droplevel(0).sort_index()
    
    def replace_dtypes(self, data, rules={'float64': 'float32', 'int32': 'int64'}):
        for k, v in rules.items():
            cols = data.dtypes[data.dtypes == k].index
            data = data.astype({t: v for t in cols}, copy=False)
        return data