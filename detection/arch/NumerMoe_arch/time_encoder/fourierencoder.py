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
    
class CFloatLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Modern PyTorch supports complex dtypes directly in nn.Linear
        self.linear = nn.Linear(in_features, out_features, dtype=torch.cfloat)

    def forward(self, x):
        return self.linear(x)
    
class LinearBlock(nn.Module):
    def __init__(self, configs, linear_name, return_mod='real'):
        super().__init__()
        self.linear_name = linear_name

        self.n_layer = configs["n_layer"]
        self.configs = configs
        self.return_mod = return_mod
        
        if linear_name == 'Real':
            self.linear_module = nn.Linear

        elif linear_name == 'Complex':
            # Use the new native complex linear layer
            self.linear_module = CFloatLinear

        elif linear_name == 'Quaternion':
            self.linear_module = QuaternionLinear

        elif linear_name == 'Octonion':
            self.linear_module = OctonionLinear

        elif linear_name == 'Sedenion':
            self.linear_module = SedenionLinear
        
        if False:
            self.linIn = nn.ModuleList([
                self._block_creator(
                                    self.configs.patch_level * self.configs.patch_dim,
                                    self.configs.d_model[0],
                                    self.linear_module)])
        else:
            self.linIn = nn.ModuleList([
                self._block_creator(
                                    self.configs['seq_len'],
                                    self.configs['d_model'][0],
                                    self.linear_module)])

        self.dropout = nn.Dropout(self.configs['dropout'])

        for i in range(1, self.n_layer):
            self.linIn.append(self._block_creator(
                                self.configs['d_model'][i-1],
                                self.configs["d_model"][i],
                                self.linear_module))
            
        # Assuming NormActivation is defined elsewhere
        self.act = NormActivation(F.tanh) 
        self.outLinear = self.linear_module(sum(self.configs["d_model"]), self.configs['pre_len'])

    def _block_creator(self, d_model, d_out, function):
        return function(d_model, d_out)

    # Use FFT for real to complex
    def real_to_complex_fft(self, x):
        # Applies Fast Fourier Transform along the last dimension
        # Output is natively a torch.cfloat tensor
        return torch.fft.fft(x, dim=-1)

    # Use IFFT for complex to real
    def complex_to_real_ifft(self, x):
        # Applies Inverse Fast Fourier Transform and extracts the real part
        return torch.fft.ifft(x, dim=-1).real

    def real_to_quaternion(self, x):
        zero = torch.zeros_like(x)  
        return torch.stack([x, zero, zero, zero], dim=-1)
        
    def real_to_octonion(self, x):
        zero = torch.zeros_like(x)
        return torch.stack([x, zero, zero, zero, zero, zero, zero, zero], dim=-1)
    
    def real_to_sedenion(self, x):
        zero = torch.zeros_like(x)  
        return torch.stack([x,    zero, zero, zero, zero, zero, zero, zero,
                            zero, zero, zero, zero, zero, zero, zero, zero], dim=-1)  
    
    def forward(self, x):
        if self.linear_name == 'Complex':
            x = self.real_to_complex_fft(x)
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
            # temp = self.dropout(temp)
            x_res[i] = temp
            x = temp

        if self.linear_name in ['Real', 'Complex']:
            x = torch.cat(x_res, dim=-1)
        else:
            x = torch.cat(x_res, dim=-2)

        x = self.outLinear(x)
        
        # Route back to real
        if self.linear_name == 'Real':
            x_real = x
        elif self.linear_name == 'Complex':
            x_real = self.complex_to_real_ifft(x)
        else:
            x_real = torch.mean(x, dim=-1)

        return x, x_real
    
class NormActivation(nn.Module):
    def __init__(self, function, eps=1e-8):
        super(NormActivation, self).__init__()
        self.func = function
        self.eps = eps
    def forward(self, x):
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True, ord=torch.inf) + self.eps
        return x / norm * self.func(norm)

class FourierEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.n_layer = configs['n_layer']
        self.norm_layer = Norm(configs['input_dim'],selected=(2,), affine=False)

        self.complex_layer = LinearBlock(configs, 'Complex')


    def forward(self, x):

        x = torch.permute(x, (0, 2, 1))
        x = self.norm_layer(x,  'norm')

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

        # stack_real = torch.cat([re_real, bi_real, qu_real],dim=-1)
        # stack_real = torch.cat([re_real, bi_real, qu_real, oc_real],dim=-1)
        # stack_real = torch.cat([re_real, bi_real],dim=-1)
        stack_real = bi_real
        # enc = self.final_fusion(stack_real)
        enc = stack_real
        return enc