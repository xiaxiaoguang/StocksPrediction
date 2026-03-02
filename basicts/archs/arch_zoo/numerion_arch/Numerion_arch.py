# from MambaModel import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from .layers import ML_Patch_Embedding,SedenionLinear,OctonionLinear,QuaternionLinear,ComplexLinear

@dataclass
class ModelArgs:
    seq_len : int
    n_layer : int
    input_dim: int
    pred_len: int
    d_model : list[int]
    dropout : int = 0.5
    patch_dim: int = 16
    patch_level: int = 4 
    device: str = 'cpu'
    def __post_init__(self):
        pass

class Norm(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, selected=(1,), affine=True, dtype=torch.float):
        super(Norm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.selected_features = selected
        self.dtype=dtype
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features,dtype=self.dtype))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features,dtype=self.dtype))

    def _get_statistics(self, x):
        dim2reduce = self.selected_features
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()

    def _normalize(self, x):
        x = x - self.mean
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        x = x + self.mean
        return x

class LinearBlock(nn.Module):
    def __init__(self, configs, linear_name, return_mod='real'):
        super().__init__()
        self.linear_name = linear_name

        self.n_layer = configs.n_layer
        self.configs=configs
        self.return_mod=return_mod
        
        if linear_name == 'Real':
            self.linear_module = nn.Linear

        elif linear_name in  ['Complex']:
            self.linear_module = ComplexLinear

        elif linear_name in ['Quaternion']:
            self.linear_module = QuaternionLinear

        elif linear_name in ['Octonion']:
            self.linear_module = OctonionLinear

        elif linear_name in ['Sedenion']:
            self.linear_module = SedenionLinear
        
        if self.configs.patch_level != -1 :
            self.linIn =nn.ModuleList([
                self._block_creator(
                                    self.configs.patch_level*self.configs.patch_dim,
                                    self.configs.d_model[0],
                                    self.linear_module)])
        else :
            self.linIn =nn.ModuleList([
                self._block_creator(
                                    self.configs.seq_len,
                                    self.configs.d_model[0],
                                    self.linear_module)])

        
        self.dropout = nn.Dropout(self.configs.dropout)

        for i in range(1,self.n_layer):
            self.linIn.append(self._block_creator(
                                self.configs.d_model[i-1],
                                self.configs.d_model[i],
                                self.linear_module))
            
        self.act = NormActivation(F.tanh)
        self.outLinear= self.linear_module(sum(self.configs.d_model) ,self.configs.pred_len)

    def _block_creator(self,d_model, d_out,function):
            return function(d_model,d_out)


    def real_to_complex(self,x):
        zero = torch.zeros_like(x)  
        return torch.stack([x, zero], dim=-1)

    def real_to_quaternion(self,x):
        zero = torch.zeros_like(x)  
        return torch.stack([x, zero, zero, zero], dim=-1)
        
    def real_to_octonion(self,x):
        zero = torch.zeros_like(x)
        return torch.stack([x, zero, zero, zero, zero, zero, zero, zero], dim=-1)
    
    def real_to_sedenion(self,x):
        zero = torch.zeros_like(x)  
        return torch.stack([x,    zero, zero, zero, zero, zero, zero, zero,
                            zero, zero, zero, zero, zero, zero, zero, zero], dim=-1)  
    
    def forward(self, x):
        if self.linear_name == 'Complex':
            x = self.real_to_complex(x)
        elif self.linear_name == 'Quaternion':
            x = self.real_to_quaternion(x)
        elif self.linear_name == 'Octonion':
            x = self.real_to_octonion(x)
        elif self.linear_name == 'Sedenion':
            x = self.real_to_sedenion(x)

        x_res = [x] * self.n_layer

        for i in range(self.n_layer):
            temp = self.linIn[i](x)    
            temp = self.act(temp)
            temp = self.dropout(temp)
            x_res[i] = temp
            x = temp

        if self.linear_name == 'Real':
            x = torch.cat(x_res,dim=-1)
        else:
            x = torch.cat(x_res,dim=-2)

        x = self.outLinear(x)
        
        if self.linear_name != 'Real':
            x_real = torch.mean(x,dim=-1)
        else:
            x_real = x
        return x,x_real

class NormActivation(nn.Module):
    def __init__(self, function, eps=1e-8):
        super(NormActivation, self).__init__()
        self.func = function
        self.eps = eps
    def forward(self, x):
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True, ord=6) + self.eps
        return x / norm * self.func(norm)

class Numerion(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.n_layer = configs.n_layer
        self.norm_layer = Norm(self.configs.input_dim,selected=(1,))

        if self.configs.patch_level != -1:
            self.multi_level_patch_embed_layer = ML_Patch_Embedding(self.configs.seq_len,self.configs.patch_dim,self.configs.patch_level,nn.Linear)

        self.real_layer = LinearBlock(configs, 'Real')

        self.complex_layer = LinearBlock(configs, 'Complex')
        
        self.quaternion_layer = LinearBlock(configs, 'Quaternion')
        
        self.octonion_layer = LinearBlock(configs, 'Octonion')
        
        self.sedenion_layer = LinearBlock(configs, 'Sedenion')

        self.gelu=nn.GELU()
        
        self.dropout= nn.Dropout(self.configs.dropout)

        self.var_fusion_layer  = nn.Linear(5,configs.pred_len)
        self.mean_fusion_layer = nn.Linear(5,configs.pred_len)
        self.output_layer_mean = nn.Linear(configs.pred_len,5)
        self.output_layer_var  = nn.Linear(configs.pred_len,5)

    def forward(self, x):
        x = self.norm_layer(x, "norm")
        x = torch.permute(x, (0, 2, 1))

        if hasattr(self,'multi_level_patch_embed_layer'): 
            x = self.multi_level_patch_embed_layer(x)

        if hasattr(self,'real_layer'):
            re,re_real = self.real_layer(x)

        if hasattr(self,'complex_layer'):
            bi,bi_real = self.complex_layer(x)
        
        if hasattr(self,'quaternion_layer'):
            qu,qu_real = self.quaternion_layer(x)

        if hasattr(self,'octonion_layer'):
            oc,oc_real = self.octonion_layer(x)

        if hasattr(self,'sedenion_layer'):
            se,se_real = self.sedenion_layer(x)

        stack_real = torch.stack([re_real, bi_real,qu_real,oc_real,se_real],dim=-1)
        
        mean_stack = self.mean_fusion_layer(stack_real)
        var_stack = self.var_fusion_layer(stack_real)
        
        act_mean = nn.functional.softmax(self.output_layer_mean(mean_stack), dim=-1)
        act_var = nn.functional.softmax(self.output_layer_var(var_stack), dim=-1)
        # breakpoint()
        mu = torch.sum(stack_real * act_mean, dim=-1)
        sigma = torch.sum(stack_real * act_var, dim=-1)
        sigma = nn.functional.softplus(sigma) + 1e-6  # Ensure positive

        mu = torch.permute(mu, (0, 2, 1))
        sigma = torch.permute(sigma, (0, 2, 1))
        mu = self.norm_layer(mu, "denorm")

        return mu,sigma
