import numpy as np
import torch
import pickle
import torch.nn.functional as F

file_path = 'Crypto/crypto_data1.npy'
Crypto = np.load(file_path)

Crypto = Crypto[:,1:,1:].astype(float)
Crypto = torch.tensor(Crypto).transpose(0,1)
Crypto = torch.flip(Crypto,dims=(0,))

labelC = torch.ones_like(Crypto)


for i in range(Crypto.shape[1]):
    j = 0
    while Crypto[j,i,0] == 0 and j < Crypto.shape[0]:
        labelC[j,i,:] = 0
        j += 1        
    
    labelC[j:j+3,i,:] = 0

    Crypto[:,i,-1]=Crypto[:,i,-2]

    for k in range(4):
        tmpj = j 
        prev = Crypto[tmpj,i,k].item()
        Crypto[tmpj,i,k] = 0 
        tmpj += 1
        while tmpj<Crypto.shape[0]:
            nxtv= Crypto[tmpj,i,k] / prev
            prev= Crypto[tmpj,i,k].item()
            Crypto[tmpj,i,k]= nxtv - 1
            tmpj += 1

Crypto = Crypto[:,:,-2].unsqueeze(-1)
labelC = labelC[:,:,1:]

file_path = 'StockD/data_in800_out16.pkl'
StockD = np.load(file_path,allow_pickle=True)
StockD = torch.tensor(StockD['processed_data'])
file_path = 'StockD/label_in800_out16.pkl'
labelD = np.load(file_path,allow_pickle=True)
labelD = torch.tensor(labelD['processed_data'])
breakpoint()

pad   = StockD.shape[0] - Crypto.shape[0]
Crypto = F.pad(Crypto, (0, 0 , 0 , 0 , pad, 0), mode='constant', value=0)
labelC = F.pad(labelC, (0, 0 , 0 , 0 , pad, 0), mode='constant', value=0)

StockD=StockD[:,:,1].unsqueeze(-1)

Data = torch.cat([StockD,Crypto],dim=1)
Label= torch.cat([labelD,labelC],dim=1)

tmpl=144
outl=12

print(Data.shape)

file_name = f'finance/data_in{tmpl}_out{outl}.pkl'
data = {'processed_data':Data.numpy()}
with open(file_name, 'wb') as file:
    pickle.dump(data, file)

print(Label.shape)

file_name = f'finance/label_in{tmpl}_out{outl}.pkl'
data = {'processed_data':Label.numpy()}
with open(file_name, 'wb') as file:
    pickle.dump(data, file)


length = Data.shape[0]-tmpl-outl
testl = int(30)
validl = int(30)
trainl = int(length)-testl-validl
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

file_name = f'finance/index_in{tmpl}_out{outl}.pkl'
with open(file_name,'wb') as file:
    pickle.dump(idx,file)
print(len(idx['train']),len(idx['test']),len(idx['valid']))

scaler = {}
scaler['func']='re_standard_transform'
scaler['args']={'mean': 0, 'std': 1}
print(scaler)
file_name = f'finance/scaler_in{tmpl}_out{outl}.pkl'
with open(file_name,'wb') as file:
    pickle.dump(scaler,file)