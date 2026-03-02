class PEMSBayArgs: # 这里换成parser就可以
    def __init__(self):
        self.device ='cuda:1'
        self.data ='PEMS-BAY' # 另一个交通数据集这里换成METR-LA
        self.root_path ='/home/srwang/survey/dataset/'
        self.data_path ='traffic/PEMS-BAY/'  #另一个交通数据集这里换成METR-LA

        self.adj_data ='/home/srwang/survey/dataset/traffic/PEMS-BAY/adj.pkl'

        self.seq_len =12 # 输入时序长度
        self.pred_len =12 # 输出时序长度

        self.batch_size =4
        self.num_nodes = 325 # 另一个换成207

        self.num_workers =0

class METRLAArgs: # 这里换成parser就可以
    def __init__(self):
        self.device ='cuda:1'
        self.data ='METR-LA' # 另一个交通数据集这里换成METR-LA
        self.root_path ='/home/srwang/survey/dataset/'
        self.data_path ='traffic/METR-LA/'  #另一个交通数据集这里换成METR-LA

        self.adj_data ='/home/srwang/survey/dataset/traffic/METR-LA/adj.pkl'

        self.seq_len =12 # 输入时序长度
        self.pred_len =12 # 输出时序长度

        self.batch_size =4
        self.num_nodes = 207 # 另一个换成207

        self.num_workers =0

class TemperatureArgs:
    def __init__(self):
        self.device ='cuda:1'
        self.data ='temperature' # 另一个气候数据这里换成cloud_cover
        self.root_path ='/home/srwang/survey/dataset/'
        self.data_path ='climate/temperature/'

        self.adj_data ='/home/srwang/survey/dataset/climate/temperature/adj.pkl'

        self.seq_len =12
        self.pred_len =12

        self.batch_size =4
        self.num_nodes = 2048 # 两个都是2048

        self.num_workers =0

class CloudCoverArgs:
    def __init__(self):
        self.device ='cuda:1'
        self.data ='cloud_cover' # 另一个气候数据这里换成cloud_cover
        self.root_path ='/home/srwang/survey/dataset/'
        self.data_path ='climate/cloud_cover/'

        self.adj_data ='/home/srwang/survey/dataset/climate/cloud_cover/adj.pkl'

        self.seq_len =12
        self.pred_len =12

        self.batch_size =4
        self.num_nodes = 2048 # 两个都是2048

        self.num_workers =0
      
class SolarEnergyArgs:
    def __init__(self):
        self.device ='cuda:1'
        self.data ='solar-energy' 
        self.root_path ='/home/srwang/survey/dataset/'
        self.data_path ='electricity/solar-energy/solar_AL.txt'

        self.adj_data ='/home/srwang/survey/dataset/electricity/solar-energy/adj.pkl'

        self.seq_len =12
        self.pred_len =12
        
        self.num_nodes = 137
        self.batch_size =64

        self.num_workers =0

class KDDCupArgs:
    def __init__(self):
        self.device ='cuda:1'
        self.data ='kddcup' 
        self.root_path ='/home/srwang/survey/dataset/'
        self.data_path ='electricity/kddcup/wtbdata_245days.csv'

        self.adj_data ='/home/srwang/survey/dataset/electricity/kddcup/adj.pkl'

        self.seq_len =12
        self.pred_len =12
        
        self.num_nodes = 134
        self.batch_size = 64

        self.num_workers = 0

class ACL18Args: # acl18
    def __init__(self):
        self.device ='cuda:1'
        self.data ='acl18' 
        self.root_path ='/home/srwang/survey/dataset/'
        self.data_path ='finance/acl18/process_data.pkl'

        self.adj_data ='/home/srwang/survey/dataset/finance/acl18/adj.pkl'

        self.seq_len =12
        self.pred_len =12
        
        self.num_nodes = 88
        self.batch_size = 64

        self.num_workers = 0
      
class KDD17Args: # kdd17
    def __init__(self):
        self.device ='cuda:1'
        self.data ='kdd17' 
        self.root_path ='/home/srwang/survey/dataset/'
        self.data_path ='finance/kdd17/data.pkl'

        self.adj_data ='/home/srwang/survey/dataset/finance/kdd17/adj.pkl'

        self.seq_len =12
        self.pred_len =12
        
        self.num_nodes = 50
        self.batch_size = 64

        self.num_workers = 0
    
class NASDAQArgs:
    def __init__(self) -> None:
        self.device ='cuda:1'
        self.data ='nasdaq' 
        self.root_path='./datasets/'
        self.seq_len=20
        self.pred_len=1
        self.num_nodes=1026
        self.num_workers=0
        self.batch_size=2
        self.use_adj='wikidata'
        self.dataset_name='nasdaq' 

class NYSEArgs:
    def __init__(self) -> None:
        self.device ='cuda:1'
        self.data ='nyse' 
        self.root_path='./datasets/'
        self.seq_len=20
        self.pred_len=1
        self.num_nodes=1737
        self.num_workers=0
        self.batch_size=2
        self.use_adj='wikidata'
        self.dataset_name='nyse' 

    
class TrafficArgs:
    def __init__(self):
        self.device ='cuda:1'
        self.data ='traffic' 
        self.root_path ='/home/srwang/survey/dataset/'
        self.data_path ='traffic/traffic/traffic.txt'

        self.adj_data ='/home/srwang/survey/dataset/traffic/traffic/adj.pkl'

        self.seq_len =96
        self.pred_len =96
        
        self.num_nodes = 1
        self.batch_size =100

        self.num_workers =0

class ElectricityArgs:
    def __init__(self):
        self.device ='cuda:1'
        self.data ='electricity' 
        self.root_path ='/home/srwang/survey/dataset/'
        self.data_path ='electricity/electricity/electricity.txt'

        self.adj_data ='/home/srwang/survey/dataset/electricity/electricity/adj.pkl'

        self.seq_len =96
        self.pred_len = 96
        
        self.num_nodes = 1
        self.batch_size =2000

        self.num_workers =0

class ExchangeRateArgs:
    def __init__(self):
        self.device ='cuda:1'
        self.data ='exchange_rate' 
        self.root_path ='/home/srwang/survey/dataset/'
        self.data_path ='finance/exchange_rate/exchange_rate.txt'

        self.adj_data ='/home/srwang/survey/dataset/finance/exchange_rate/adj.pkl'

        self.seq_len =12
        self.pred_len =12
        
        self.num_nodes = 1
        self.batch_size =2000

        self.num_workers =0

class ETTh1Args:
    def __init__(self):
        self.device ='cuda:1'
        self.data ='ETTh1' 
        self.root_path ='/home/srwang/survey/dataset/'
        self.data_path ='electricity/ETT/ETTh1.csv'

        self.adj_data ='/home/srwang/survey/dataset/electricity/ETT/adj.pkl'

        self.seq_len =96
        self.pred_len =96
        
        self.num_nodes = 1
        self.batch_size =2000

        self.num_workers =0

class ETTh2Args:
    def __init__(self):
        self.device ='cuda:1'
        self.data ='ETTh2' 
        self.root_path ='/home/srwang/survey/dataset/'
        self.data_path ='electricity/ETT/ETTh2.csv'

        self.adj_data ='/home/srwang/survey/dataset/electricity/ETT/adj.pkl'

        self.seq_len =96
        self.pred_len =96
        
        self.num_nodes = 1
        self.batch_size =2000

        self.num_workers =0

class ETTm1Args:
    def __init__(self):
        self.device ='cuda:1'
        self.data ='ETTm1' 
        self.root_path ='/home/srwang/survey/dataset/'
        self.data_path ='electricity/ETT/ETTm1.csv'

        self.adj_data ='/home/srwang/survey/dataset/electricity/ETT/adj.pkl'

        self.seq_len =48
        self.pred_len =48
        
        self.num_nodes = 1
        self.batch_size =2000

        self.num_workers =0

class ETTm2Args:
    def __init__(self):
        self.device ='cuda:1'
        self.data ='ETTm2' 
        self.root_path ='/home/srwang/survey/dataset/'
        self.data_path ='electricity/ETT/ETTm2.csv'

        self.adj_data ='/home/srwang/survey/dataset/electricity/ETT/adj.pkl'

        self.seq_len =96
        self.pred_len =96
        
        self.num_nodes = 1
        self.batch_size =2000

        self.num_workers =0

def load_arg(name):
    if name=='pems':
        return PEMSBayArgs()
    elif name=='metr':
        return METRLAArgs()
    elif name=='temp':
        return TemperatureArgs()
    elif name=='cloud':
        return CloudCoverArgs()
    elif name=='solar':
        return SolarEnergyArgs()
    elif name=='kddcup':
        return KDDCupArgs()
    elif name=='acl':
        return ACL18Args()
    elif name=='kdd17':
        return KDD17Args()
    elif name=='traffic':
        return TrafficArgs()
    elif name=='electricity':
        return ElectricityArgs()
    elif name=='exchange_rate':
        return ExchangeRateArgs()
    elif name=='ETTh1':
        return ETTh1Args()
    elif name=='ETTh2':
        return ETTh2Args()
    elif name=='ETTm1':
        return ETTm1Args()
    elif name=='ETTm2':
        return ETTm2Args()
    elif name=='nasdaq':
        return NASDAQArgs()
    elif name=='nyse':
        return NYSEArgs()
    else:
        raise NotImplementedError(f'dataset {name} not exist.')