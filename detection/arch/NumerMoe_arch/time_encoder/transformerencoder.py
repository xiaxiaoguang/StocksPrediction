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
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + 1e-5).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        x = (x * self.stdev + self.mean)
        return x

class LinearBlock(nn.Module):
    def __init__(self, configs, linear_name, return_mod='real'):
        super().__init__()
        self.linear_name = linear_name

        self.n_layer = configs["n_layer"]
        self.configs = configs
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
        
        # if self.configs.patch_level != -1 :
        if False:
            self.linIn =nn.ModuleList([
                self._block_creator(
                                    self.configs.patch_level*self.configs.patch_dim,
                                    self.configs.d_model[0],
                                    self.linear_module)])
        else :
            self.linIn =nn.ModuleList([
                self._block_creator(
                                    self.configs['seq_len'],
                                    self.configs['d_model'][0],
                                    self.linear_module)])

        
        self.dropout = nn.Dropout(self.configs['dropout'])

        for i in range(1,self.n_layer):
            self.linIn.append(self._block_creator(
                                self.configs['d_model'][i-1],
                                self.configs["d_model"][i],
                                self.linear_module))
            
        self.act = NormActivation(F.tanh)
        self.outLinear= self.linear_module(sum(self.configs["d_model"]) , self.configs['pre_len'])

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
            # temp = self.act(temp)
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
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True, ord=torch.inf) + self.eps
        return x / norm * self.func(norm)

class LinearTransformerEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.n_layer = configs.get('n_layer', 1)
        
        # Assuming Norm is a custom class defined elsewhere in your project
        self.norm_layer = Norm(configs['input_dim'], selected=(2,), affine=False)
        print("Initializing Ablation: Linear + Transformer Encoder")
        
        # Linear projection block
        self.real_layer = LinearBlock(configs, 'Real')
        
        # Setup Transformer with best default parameters for ablations
        # Assuming configs['pre_len'] maps to the feature dimension (d_model) out of the LinearBlock
        d_model = configs.get('pre_len', 512) 
        nhead = configs.get('nhead', 8)
        num_layers = configs.get('trans_layers', 2) # 2-3 layers is usually a good default for an ablation
        dim_feedforward = configs.get('dim_feedforward', d_model * 4) # Standard 4x expansion
        dropout = configs.get('dropout', 0.1)

        # Using PyTorch's native Transformer layer for optimized performance
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation='gelu', # GeLU generally outperforms ReLU in modern transformers
            batch_first=True   # Ensures input format is (batch, seq, feature)
        )
        self.trans = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # Initial permutation
        x = torch.permute(x, (0, 2, 1))
        
        # Normalization
        x = self.norm_layer(x, 'norm')

        # Linear encoding phase
        if hasattr(self, 'real_layer'):
            re, re_real = self.real_layer(x)
        else:
            re_real = x

        # Transformer encoding phase
        # Note: re_real must be of shape (batch_size, sequence_length, d_model) 
        # to work natively with batch_first=True
        enc = self.trans(re_real)

        return enc