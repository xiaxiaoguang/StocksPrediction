import sys
sys.path.append('/home/wbyan/stock_data/')

import numpy as np

import pickle
from data_utils import load_data #会显示报错，但是上面加了sys path可以使用
#读取股市数据
name = "csi500"

date_list, features, labels, stock_list = load_data(
        data_type="ch_daily_processed",
        stock_list=name,
        start_date="2005-01-01",
        end_date="2025-01-01",
        contain_bj=False,
        drop_short=False,
        short_len=1000,
)
tmpl=96
outl=3
selectk=500
features = features.permute(1,0,2)
labels = labels.permute(1,0,2)


# import torch
# indix1 = torch.where(features[0,:,0]!=0,1,0)
# indix2 = torch.where(features[-1,:,0]!=0,1,0)
# indix = (indix1*indix2).nonzero().flatten()
# print(indix.shape)
# print(features[0,indix,1][:100])
# print(features[0,indix,12])
# tmp = torch.sum(features[:,indix,12],dim=0)
# print(tmp.shape)
# index = torch.topk(tmp,k=selectk)[1].flatten()
# print(indix[index])
# print(features.shape)
# feature = features[:,indix[index],:]
# labels = labels[:,indix[index],:]

labels  = labels.numpy()
try :
    features=feature.numpy()
except:
    print("no selection and clean")
    features=features.numpy()
import os
if not os.path.exists(f"{name}/"):
    os.mkdir(f"{name}/")
file_name = f'{name}/data_in{tmpl}_out{outl}.pkl'
data = {'processed_data':features}
with open(file_name, 'wb') as file:
    pickle.dump(data, file)
file_name = f'{name}/label_in{tmpl}_out{outl}.pkl'
data = {'processed_data':labels}
with open(file_name, 'wb') as file:
    pickle.dump(data, file)
# tmpl = 144
length = features.shape[0]-tmpl-outl
testl = int(245)
validl = int(245)
trainl = int(length)-testl-validl
# trainl = 8
# testl = 8
# validl = 8
# print(trainl,testl,validl)

idx = {}
train = []
for i in range(0,trainl,1):
    train.append((i,i+tmpl,i+tmpl+outl))
idx['train']=train
valid=[]
for i in range(trainl,trainl+testl,1):
    valid.append((i,i+tmpl,i+tmpl+outl))
idx['valid']=valid
test=[]
for i in range(trainl+testl,trainl+testl+validl,1):
    test.append((i,i+tmpl,i+tmpl+outl))
idx['test'] =test

file_name = f'{name}/index_in{tmpl}_out{outl}.pkl'

with open(file_name,'wb') as file:
    pickle.dump(idx,file)

# print(train)
print(len(idx['train']),len(idx['test']))
scaler = {}
scaler['func']='re_standard_transform'
scaler['args']={'mean': 0, 'std': 1}

file_name = f'{name}/scaler_in{tmpl}_out{outl}.pkl'

with open(file_name,'wb') as file:
    pickle.dump(scaler,file)
file_name = f'{name}/scaler_in{tmpl}_out{outl}.pkl'
with open(file_name,'rb') as file:
    tmp =  pickle.load(file)
# print(tmp)
print(len(train),testl,validl)