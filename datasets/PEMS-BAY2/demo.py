import sys
sys.path.append('/home/srwang/survey')
from dataprovider.data_factory import data_provider
from dataprovider.data_factory import load_adj
sys.path.append('/home/srwang/WaveGraph/dataprovider')
from data_factory import data_provider as data_provider_nasnys
from data_factory import load_adj as load_adj_nasnys
import argparse
from dataArgs import *
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--device",default=0,type=int)

parser.add_argument("--use_adj",default=True,type=bool)

args = parser.parse_args()

def load_data():
  device=f'cuda:{args.device}'
  
  if args.dataset_name in  ['nasdaq','nyse']:
        data_args=load_arg(name=args.dataset_name)
        data_args.device=device
        
        train_dataset, train_dataloader = data_provider_nasnys(data_args, 'train')
        val_dataset, val_dataloader = data_provider_nasnys(data_args, 'val')
        test_dataset, test_dataloader = data_provider_nasnys(data_args, 'test')

        adj, mask = load_adj_nasnys(market_name=data_args.data, relation_name=data_args.use_adj) 
        adj=np.sum(adj,axis=-1).astype(np.float64)
        adj[adj>1]=1

  else:
        data_args=load_arg(name=args.dataset_name)
        data_args.device=device

        train_dataset, train_dataloader = data_provider(data_args, 'train')
        val_dataset, val_dataloader = data_provider(data_args, 'val')
        test_dataset, test_dataloader = data_provider(data_args, 'test')

        adj = load_adj(data_args.data, data_args.adj_data)