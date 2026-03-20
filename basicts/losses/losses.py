import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ..utils import check_nan_inf


def l1_loss(input_data, target_data, **kwargs):
    """unmasked mae."""

    return F.l1_loss(input_data, target_data)


def l2_loss(input_data, target_data, **kwargs):
    """unmasked mse"""

    check_nan_inf(input_data)
    check_nan_inf(target_data)
    return F.mse_loss(input_data, target_data)


def returnLAM(prediction,real_value ,null_val: float = np.nan,M5 = 2):
    # L1 loss and topk margin
    if prediction.shape[3]!=1:
        prediction=prediction[:,:,:,1]
    if real_value.shape[3]!=1:
        real_value=real_value[:,:,:,1].unsqueeze(-1)
    if prediction.shape != real_value.shape:
        prediction = prediction.squeeze(-1)
    prediction_return = torch.prod(torch.add(prediction,1),dim=1)
    real_return = torch.prod(torch.add(real_value,1),dim=1)
    # dif1  = torch.nn.functional.l1_loss(prediction_return,real_return)
    tmp1 = torch.abs(torch.topk(prediction_return,k=M5,dim=1)[0]-
                     torch.topk(real_return,k=M5,dim=1)[0])
    tmp2 = torch.abs(torch.topk(prediction_return,k=M5,dim=1,largest=False)[0]-
                     torch.topk(real_return,k=M5,dim=1,largest=False)[0])
    # tmp += torch.abs(torch.topk(prediction_return,largest=False,k=M5,dim=1)[0]-torch.topk(real_return,largest=False,k=M5,dim=1)[0])
    dif2 = torch.mean(tmp1+tmp2)
    return  dif2

def normalized(prediction):
    MX = torch.max(prediction,dim=1,keepdim=True)[0]
    MI = torch.min(prediction,dim=1,keepdim=True)[0]
    return (prediction-MI)/(MX-MI)

def PLKLoss(prediction,real_value,null_val: float = np.nan):
    if prediction.shape[3]!=1:
        prediction=prediction[:,:,:,1]
    if real_value.shape[3]!=1:
        real_value=real_value[:,:,:,1].unsqueeze(-1)
    if prediction.shape != real_value.shape:
        prediction = prediction.squeeze(-1)
    prediction_return = torch.prod(torch.add(prediction,1),dim=1)
    real_return = torch.prod(torch.add(real_value,1),dim=1)
    B = prediction.shape[0]
    N = prediction_return.shape[1]
    sort_index = torch.argsort(real_return,dim=1,descending=True).view(-1)
    indices_array = torch.repeat_interleave((torch.arange(B)), N)
    a = prediction_return[indices_array,sort_index]
    tmp = ((torch.arange(start=-N,end=N,step=2,device=prediction_return.device).unsqueeze(-1).repeat(B,1) / N))
    a = torch.sum(a * tmp) / B
    return a

def construct_unimodal_function(N, l, device):
    x = torch.linspace(0, N, steps=N,device=device)
    sigma = (N / 10)  # Standard deviation, adjust to control the width of the peak
    y = torch.exp(-0.5 * ((x - l) / sigma) ** 2)
    return x, y

def generate_sequence(N, L , gap , device):
    if not (0 <= L <= N):
        raise ValueError("L should be in the range [0, N]")
    sequence = torch.zeros(N,device=device)

    if L > 0:
        sequence[:L] = (torch.arange(L + 1, 1 , step=-1).float())/2/(L+1)+gap
    if L < N:
        sequence[L:] = (-torch.arange(1, N - L + 1).float())/2/(N-L+1)-gap

    return sequence

def MDLoss(prediction,real_value, topk=30 ,null_val: float = np.nan):
    if prediction.shape[3]!=1:
        prediction=prediction[:,:,:,1]
    if real_value.shape[3]!=1:
        real_value=real_value[:,:,:,1].unsqueeze(-1)
    if prediction.shape != real_value.shape:
        prediction = prediction.squeeze(-1)
    prediction_return = torch.prod(torch.add(prediction,1),dim=1)
    real_return = torch.prod(torch.add(real_value,1),dim=1)

    B = prediction.shape[0]
    N = prediction_return.shape[1]
    sort_index = torch.argsort(real_return,dim=1,descending=True)
    # sort_index = sort_index.view(-1)
    # indices_array = torch.repeat_interleave((torch.arange(B)), N)
#real_return.gather(1, sort_index[:, :, 0].unsqueeze(-1))
    sorted_pred = torch.exp(prediction_return.gather(1, sort_index[:, :, 0].unsqueeze(-1)).squeeze(-1))
    # Sum the topk and remaining elements along the specified dimension
    topk_sum = (torch.log(sorted_pred[:, :topk])).sum(dim=-1)
    rest_sum = -torch.log(torch.cumsum(torch.flip(sorted_pred,[1]),dim=1))[:,-topk:].sum(dim=-1)
    
    # Compute the loss
    loss = (topk_sum+rest_sum).sum() / (B)
    # PLKLoss = torch.mean(-torch.log(prediction_return[indices_array,sort_index]))
    # PLKLoss += torch.mean(torch.log(torch.cumsum(prediction_return[indices_array,sort_index], dim=0)))
    return loss


def MRLoss(prediction,real_value, margin=0.02 , gap=0.5 ,null_val: float = np.nan):
    if prediction.shape[3]!=1:
        prediction=prediction[:,:,:,1]
    if real_value.shape[3]!=1:
        real_value=real_value[:,:,:,1].unsqueeze(-1)
    if prediction.shape != real_value.shape:
        prediction = prediction.squeeze(-1)

    prediction_return = torch.prod(torch.add(prediction,1),dim=1)
    real_return = torch.prod(torch.add(real_value,1),dim=1)

    B = prediction.shape[0]
    N = prediction_return.shape[1]
    
    sort_index = torch.argsort(real_return,dim=1,descending=True)
    # indices_array = torch.repeat_interleave((torch.arange(B)), N)

    # tmp = ((torch.arange(start=N-N/maxk,end=-N/maxk,step=-1,device=prediction_return.device).unsqueeze(-1)/N - 0.5) * 2)
    # # tmp[-(maxk+maxk):,:] -= torch.sigmoid(torch.arange(start=-maxk,end=maxk,step=1,device=prediction_return.device)).unsqueeze(-1)
    # tmp = torch.exp(tmp)-torch.exp(-tmp)
    # tmp = tmp.repeat(B,1)
    MRLoss = torch.nn.MarginRankingLoss(margin)
    loss = 0
    for i in range(B):
        a = prediction_return[i,sort_index[i,:,0],0]
        b = real_return [i,sort_index[i,:,0],0]
        _p = (b <= 1).nonzero()
        if _p.shape[0] == 0:
            piv = N-1
        else :
            piv = _p[0].item()
        tmp = generate_sequence(N,piv,gap,device=prediction_return.device)
        # loss += (tmp*(b-a)).sum()/N
        loss += MRLoss(a,b,tmp)
    loss /= B
    return loss

def MRLoss2(prediction,real_value, margin=0.02 , maxk=10 ,null_val: float = np.nan):
    if prediction.shape[3]!=1:
        prediction=prediction[:,:,:,1]
    if real_value.shape[3]!=1:
        real_value=real_value[:,:,:,1].unsqueeze(-1)
    if prediction.shape != real_value.shape:
        prediction = prediction.squeeze(-1)

    prediction_return = torch.prod(torch.add(prediction,1),dim=1)
    real_return = torch.prod(torch.add(real_value,1),dim=1)

    B = prediction.shape[0]
    N = prediction_return.shape[1]
    
    sort_index = torch.argsort(real_return,dim=1,descending=True).view(-1)
    indices_array = torch.repeat_interleave((torch.arange(B)), N)

    tmp = ((torch.arange(start=N-N/maxk,end=-N/maxk,step=-1,device=prediction_return.device).unsqueeze(-1)/N - 0.5) * 2)
    # tmp[-(maxk+maxk):,:] -= torch.sigmoid(torch.arange(start=-maxk,end=maxk,step=1,device=prediction_return.device)).unsqueeze(-1)
    tmp = torch.exp(tmp)-torch.exp(-tmp)
    tmp = tmp.repeat(B,1)

    a = prediction_return[indices_array,sort_index]
    b = real_return[indices_array,sort_index]
    MRLoss = torch.nn.MarginRankingLoss(margin)
    loss = MRLoss(a,b,tmp)
    return loss

def PLKnce(hidden_space, sort_index , topk=20 , null_val:float = np.nan):
    B,N,E = hidden_space.shape
    sort_index = sort_index.repeat(1,1,E)
    sorted_pred = hidden_space.gather(1, sort_index).squeeze(-1)
    
    similarity = torch.exp(F.cosine_similarity(sorted_pred[:,0].unsqueeze(1), sorted_pred[:,:],dim=-1))
    topk_sum = -(torch.log(similarity[:, :topk])).sum(dim=-1)
    rest_sum = torch.log(torch.cumsum(torch.flip(similarity[:,topk:],[1]),dim=1))[:,-topk:].sum(dim=-1)
    loss = (topk_sum+rest_sum).sum() / (B)

    # sort_index = torch.flip(sort_index,[1])
    # sorted_pred = hidden_space.gather(1, sort_index).squeeze(-1)
    # similarity = torch.exp(F.cosine_similarity(sorted_pred[:,0].unsqueeze(1), sorted_pred[:,:],dim=-1))
    # topk_sum = -(torch.log(similarity[:, :topk])).sum(dim=-1)
    # rest_sum = torch.log(torch.cumsum(torch.flip(similarity[:,topk:],[1]),dim=1))[:,-topk:].sum(dim=-1)
    # loss += (topk_sum+rest_sum).sum() / (B)

    return loss

def RightOrder(prediction,real_value,null_val: float = np.nan):
    # 这应该有一个置信度分数才对
    if prediction.shape[3]!=1:
        prediction=prediction[:,:,:,1]
    if real_value.shape[3]!=1:
        real_value=real_value[:,:,:,1].unsqueeze(-1)
    if prediction.shape != real_value.shape:
        prediction = prediction.squeeze(-1)
    prediction_return = torch.sigmoid(torch.add(torch.prod(torch.add(prediction,1),dim=1),-1))
    real_return = torch.prod(torch.add(real_value,1),dim=1)
    target = torch.where(real_return > 1.0,1.0,0.0)
    # prediction_return = torch.where(prediction_return > 1.0 , torch.ones_like(prediction_return), prediction_return)
    # prediction_return = torch.where(prediction_return < 0.0 , torch.zeros_like(prediction_return), prediction_return)
    return nn.functional.binary_cross_entropy(prediction_return,target)
