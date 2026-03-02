import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat,rearrange

def Contrastive(hidden_state_origin,hidden_state_repeat,prob=None):
    # You shouldn't change these
    R,B,T,E = hidden_state_origin.shape
    _,_,TK,_ = hidden_state_repeat.shape
    assert TK % T ==0 
    K = TK/T
    device=hidden_state_origin.device
    index_b_01 = torch.arange(start=1,end=T,dtype=torch.long,device=device)
    index_b_0 = torch.arange(start=K,end=TK,step=K,dtype=torch.long,device=device)    
    index_b_1 = torch.arange(start=K-1,end=TK-K,step=K,dtype=torch.long,device=device)
    # index_b_2 = torch.arange(start=K-2,end=TK-K,step=K,dtype=torch.long,device=device)
    index_b_3 = torch.arange(end=T-1,dtype=torch.long,device=device)
    index_b_4 = torch.arange(start=0,end=TK-K,step=K,dtype=torch.long,device=device)    
    # until here
    # --------------MENTION !!!!!!!-----------------
    i = 0
    # for i in range(R-1,R):
    # MENTION THERE
    query = torch.index_select(hidden_state_repeat[i],dim=1,index=index_b_1)
    # HERE is what you should make a change, now you can say its cosine similarity with -logExp as infoNCE loss
    # YOU shouldn't change the positive / negative pairs it self, but change the method of calculate loss
    positive = torch.exp(F.cosine_similarity(query, torch.index_select(hidden_state_origin[i],dim=1,index=index_b_3) ,dim=-1))
    
    # positive2 = torch.exp(F.cosine_similarity(query, torch.index_select(hidden_state_repeat[i],dim=1,index=index_b_2),dim=-1))
    
    positive3 = torch.exp(F.cosine_similarity(query, torch.index_select(hidden_state_repeat[i],dim=1,index=index_b_4),dim=-1))

    negative = torch.exp(F.cosine_similarity(query, torch.index_select(hidden_state_repeat[i],dim=1,index=index_b_0) ,dim=-1))
    
    negative1 = torch.exp(F.cosine_similarity(query, torch.index_select(hidden_state_origin[i],dim=1,index=index_b_01) ,dim=-1))

    # if prob is None:
    #     prob = (1.5 ** nn.Parameter(torch.arange(1, T, dtype=torch.float32,device='cuda')))
    #     prob /= prob.sum()
    # prob = repeat(prob,'L -> B L',B=negative.shape[0])
    loss = torch.mean(torch.sum(-torch.log((positive +  positive3) / ( positive + positive3 + negative + negative1)),dim=1))
    return loss


def Contrastive0(hidden_state_origin,hidden_state_repeat,prob=None):
    R,T,E = hidden_state_origin.shape
    _,TK,_ = hidden_state_repeat.shape
    assert TK % T ==0 
    K = TK/T
    device=hidden_state_origin.device
    index_b_01 = torch.arange(start=1,end=T,dtype=torch.long,device=device)
    index_b_0 = torch.arange(start=K,end=TK,step=K,dtype=torch.long,device=device)    
    index_b_1 = torch.arange(start=K-1,end=TK-K,step=K,dtype=torch.long,device=device)
    index_b_3 = torch.arange(end=T-1,dtype=torch.long,device=device)
    index_b_4 = torch.arange(start=0,end=TK-K,step=K,dtype=torch.long,device=device)    

    query = torch.index_select(hidden_state_repeat,dim=1,index=index_b_1)
    positive = torch.exp(F.cosine_similarity(query, torch.index_select(hidden_state_origin,dim=1,index=index_b_3) ,dim=-1))
    positive3 = torch.exp(F.cosine_similarity(query, torch.index_select(hidden_state_repeat,dim=1,index=index_b_4),dim=-1))
    negative = torch.exp(F.cosine_similarity(query, torch.index_select(hidden_state_repeat,dim=1,index=index_b_0) ,dim=-1))
    negative1 = torch.exp(F.cosine_similarity(query, torch.index_select(hidden_state_origin,dim=1,index=index_b_01) ,dim=-1))
    # breakpoint()
    loss = torch.mean(torch.sum(-torch.log((positive + positive3) / (positive3 + positive + negative + negative1)),dim=1))
    return loss

def Contrastive2(hidden_state_origin,hidden_state_repeat):
    B,T,N,E = hidden_state_origin.shape
    hidden_state_origin = rearrange(hidden_state_origin,'a b c d->(a c) b d')
    _,TK,_ = hidden_state_repeat.shape
    assert TK % T ==0 
    K = TK/T

    device=hidden_state_origin.device
    index_b_01 = torch.arange(start=1,end=T,dtype=torch.long,device=device)
    index_b_0 = torch.arange(start=K,end=TK,step=K,dtype=torch.long,device=device)    
    index_b_1 = torch.arange(start=K-1,end=TK-K,step=K,dtype=torch.long,device=device)
    index_b_2 = torch.arange(start=K-2,end=TK-K,step=K,dtype=torch.long,device=device)
    index_b_3 = torch.arange(end=T-1,dtype=torch.long,device=device)
    index_b_4 = torch.arange(start=0,end=TK-K,step=K,dtype=torch.long,device=device)  

    Loss  = 0
    query = torch.index_select(hidden_state_repeat,dim=1,index=index_b_1)

    positive1 = F.pairwise_distance(query, torch.index_select(hidden_state_origin,dim=1,index=index_b_3), p=1)
    positive2 = F.pairwise_distance(query, torch.index_select(hidden_state_repeat,dim=1,index=index_b_2), p=1)
    positive3 = F.pairwise_distance(query, torch.index_select(hidden_state_repeat,dim=1,index=index_b_4), p=1)

    negative1 = F.pairwise_distance(query, torch.index_select(hidden_state_repeat,dim=1,index=index_b_0), p=1)
    negative2 = F.pairwise_distance(query, torch.index_select(hidden_state_origin,dim=1,index=index_b_01),p=1)
    breakpoint()
    Loss += torch.mean(torch.sum(torch.clamp(positive1 + positive2 + positive3 - 2 * negative1 - negative2 ,min=0),dim=-1))

    return Loss



def Contrastive3(hidden_state_origin, hidden_state_repeat):
    R, B, T, E = hidden_state_origin.shape
    _, _, TK, _ = hidden_state_repeat.shape
    assert TK % T == 0
    K = TK // T
    device=hidden_state_origin.device
    index_b_01 = torch.arange(start=1,end=T,dtype=torch.long,device=device)
    index_b_0 = torch.arange(start=K,end=TK,step=K,dtype=torch.long,device=device)    
    index_b_1 = torch.arange(start=K-1,end=TK-K,step=K,dtype=torch.long,device=device)
    index_b_2 = torch.arange(start=K-2,end=TK-K,step=K,dtype=torch.long,device=device)
    index_b_3 = torch.arange(end=T-1,dtype=torch.long,device=device)
    index_b_4 = torch.arange(start=0,end=TK-K,step=K,dtype=torch.long,device=device)  
    Loss = 0

        # query = hidden_state_repeat[i,grid_a, grid_b_1, grid_c]
        # positive1 = F.pairwise_distance(query,hidden_state_origin[i,grid_a, grid_b_3, grid_c])
        # positive2 = F.pairwise_distance(query,hidden_state_repeat[i,grid_a, grid_b_2, grid_c])
        # positive3 = F.pairwise_distance(query,hidden_state_repeat[i,grid_a, grid_b_4,grid_c])
        # negative1 = F.pairwise_distance(query,hidden_state_repeat[i,grid_a,grid_b_0,grid_c])        
        # negative2 = F.pairwise_distance(query,hidden_state_origin[i,grid_a,grid_b_01,grid_c])  
    for i in range(R - 1):  # every layers
        query = torch.index_select(hidden_state_repeat[i],dim=1,index=index_b_1)
        # Calculate the positive and negative similarities as probabilities
        positive_prob = F.log_softmax(F.cosine_similarity(
            query, torch.index_select(hidden_state_origin[i],dim=1,index=index_b_3), dim=-1), dim=-1)
        positive_prob2 = F.log_softmax(F.cosine_similarity(
            query, torch.index_select(hidden_state_repeat[i],dim=1,index=index_b_2), dim=-1), dim=-1)
        positive_prob3 = F.log_softmax(F.cosine_similarity(
            query, torch.index_select(hidden_state_repeat[i],dim=1,index=index_b_4), dim=-1), dim=-1)
        negative_prob = F.softmax(F.cosine_similarity(
            query, torch.index_select(hidden_state_repeat[i],dim=1,index=index_b_0), dim=-1), dim=-1)
        negative_prob1 = F.softmax(F.cosine_similarity(
            query, torch.index_select(hidden_state_origin[i],dim=1,index=index_b_01), dim=-1), dim=-1)
        # Aggregate positive probabilities
        positive_combined = positive_prob + positive_prob2 + positive_prob3

        # Aggregate negative probabilities
        negative_combined = negative_prob + negative_prob1
        kl_div = nn.KLDivLoss()(
            positive_combined, 
            negative_combined,
        )
        Loss += kl_div

    return Loss


