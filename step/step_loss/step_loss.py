import numpy as np
import torch
from torch import nn
from basicts.losses import *

def step_loss(prediction, real_value, theta, priori_adj, gsl_coefficient, epoch ,null_val=np.nan):
    # graph structure learning loss
    # prediction/realvalue: torch.Size([32, 12, 207, 1]),  theta : torch.Size([32, 207, 207])
    # priori_adj torch.Size([32, 207, 207]) gsl_coefficient is a float
    B, N, N = theta.shape
    theta = theta.view(B, N*N)
    tru = priori_adj.view(B, N*N)
    BCE_loss = nn.BCELoss()
    loss_graph = BCE_loss(theta, tru)
    loss_pred = masked_mae(prediction, real_value, null_val=null_val)
    # loss_returnLAM = returnLAM(prediction=prediction, real_value=real_value, null_val=null_val)
    loss_MR = MRLoss(prediction=prediction, real_value=real_value)
    # loss_MR = 0
    # loss_MD = MDLoss(prediction=prediction, real_value=real_value)
    # loss_MD = MDLoss(prediction=prediction, real_value=real_value)
    # loss_PLKLoss = PLKLoss(prediction=prediction,real_value=real_value)
    loss = loss_pred + loss_graph * gsl_coefficient +  loss_MR
    # + loss_returnLAM +  + 
    return loss

def step4_loss(prediction, real_value, theta, priori_adj, gsl_coefficient, epoch ,null_val=np.nan):
    # prediction loss
    loss_pred = masked_mae(prediction, real_value, null_val=null_val)
    # loss_returnLAM = returnLAM(prediction=prediction, real_value=real_value, null_val=null_val)
    loss_MR = MRLoss(prediction=prediction, real_value=real_value)
    loss_MD = MDLoss(prediction=prediction, real_value=real_value)
    loss = loss_pred * 10  + loss_MR + loss_MD * 5
    #+ loss_MR + loss_MD + loss_returnLAM
    return loss


def stock_loss(prediction, real_value, theta, priori_adj, gsl_coefficient, epoch , null_val=np.nan):
    # graph structure learning loss
    # prediction/realvalue: torch.Size([32, 12, 207, 1]),  theta : torch.Size([32, 207, 207])
    # priori_adj torch.Size([32, 207, 207]) gsl_coefficient is a float
    B, N, N = theta.shape
    theta = theta.view(B, N*N)
    tru = priori_adj.view(B, N*N)
    BCE_loss = nn.BCELoss()
    loss_graph = BCE_loss(theta, tru)
    # prediction loss
    # loss_MD = MDLoss(prediction=prediction, real_value=real_value)
    loss_MR = MRLoss(prediction=prediction, real_value=real_value)
    loss = loss_graph * gsl_coefficient + loss_MR 

    return loss

def tsformer_loss(prediction, real_value, null_val=np.nan):
    loss_pred = masked_mae(preds=prediction, labels=real_value, null_val=null_val)
    loss_pred += masked_mape(preds=prediction, labels=real_value, null_val=null_val)
    loss_pred += masked_rmse(preds=prediction, labels=real_value, null_val=null_val)
    return loss_pred

def caoformer_loss(predict_future, real_future, hidden_time, alongTime,hidden_space,alongSpace, epoch = 0,null_val=np.nan):
    loss_1 = masked_mae(predict_future,real_future)+masked_rmse(predict_future,real_future)
    loss_2_ti = PLKnce(hidden_time,alongTime)
    loss_2_sp = PLKnce(hidden_space,alongSpace)
    if epoch % 200 == 0:
        print(loss_2_ti,loss_2_sp)
    return loss_1 + loss_2_ti + loss_2_sp

def smtsf_loss(prediction1, real_value1,prediction2,real_value2, null_val=np.nan):
    loss_pred = tsformer_loss(prediction1,real_value1) + tsformer_loss(prediction2,real_value2)
    return loss_pred

def normal_loss(prediction, real_value, null_val=0.0):
    loss_pred = masked_mae(preds=prediction,labels=real_value, null_val=null_val)
    return loss_pred

def normal_Stock(prediction, real_value, null_val=np.nan):
    loss_pred = masked_mae(preds=prediction,labels=real_value, null_val=null_val)
    loss_MR = MRLoss(prediction=prediction, real_value=real_value)
    loss = loss_pred + loss_MR
    return loss

def mamba_loss(prediction,real_value, hso,hsr, prob=None,null_val=np.nan):
    loss_pred = masked_mae(preds=prediction, labels=real_value, null_val=null_val)
    loss_cons = Contrastive(hidden_state_origin=hso,hidden_state_repeat=hsr,prob=prob)
    # breakpoint()
    return loss_pred + loss_cons

def mamba_loss2(prediction,real_value, hso,hsr, prob=None,null_val=np.nan):
    loss_pred = masked_mae(preds=prediction, labels=real_value, null_val=null_val)
    loss_cons = Contrastive0(hidden_state_origin=hso,hidden_state_repeat=hsr,prob=prob)
    return loss_pred + loss_cons

def mamba_KL(prediction,real_value,hso,hsr,null_val=np.nan):
    loss_pred = masked_mae(preds=prediction, labels=real_value, null_val=null_val)
    loss_cons = Contrastive2(hidden_state_origin=hso,hidden_state_repeat=hsr)
    return loss_pred + loss_cons
