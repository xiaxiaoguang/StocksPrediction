import torch
from torch import nn
from timm.models.vision_transformer import trunc_normal_
from .patch import PatchEmbedding
from .mask import MaskGenerator
from .positional_encoding import PositionalEncoding
from .transformer_layers import TransformerLayers,TransformerLayers2
from .sampling import Upsampling,Downsampling,Upsummary

def unshuffle(shuffled_tokens):
    dic = {}
    for k, v, in enumerate(shuffled_tokens):
        dic[v] = k
    unshuffle_index = []
    for i in range(len(shuffled_tokens)):
        unshuffle_index.append(dic[i])
    return unshuffle_index

class MSCFormer(nn.Module):
    """An efficient unsupervised pre-training model for Time Series based on transFormer blocks. (TSFormer)"""

    def __init__(self,patch , sample_channel,
                 embed_dim, num_heads, mlp_ratio,
                 dropout, num_tokens, mask_ratio,num_feats,
                 num_decoder,decoder_depth, selected_feature,
                 mode="pre-train"):
        super().__init__()
        assert mode in ["pre-train", "forecasting"], "Error mode."
        # assert time in ["week","month"],"Error version"
        
        self.patch = patch
        self.patch_size = 1
        for i in self.patch:
            self.patch_size = self.patch_size * i
            
        self.num_token=num_tokens//self.patch_size
        self.num_decoder = num_decoder
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mask_ratio = mask_ratio
        self.mode = mode
        self.mlp_ratio = mlp_ratio
        self.selected_feature = selected_feature

        # self.encoder_norm = nn.LayerNorm(embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim , patch , dropout=dropout) 

        self.patch_embedding = PatchEmbedding(patch, embed_dim ,
                                              norm_layer=None, num_feats=num_feats)
        
        self.device = next(self.parameters()).device    

        self.mask = MaskGenerator(self.num_token ,patch , mask_ratio , self.device)
        
        self.up_summary = Upsummary(patch,num_decoder,embed_dim)
        self.up_sampling = Upsampling(patch,num_decoder,sample_channel)
        self.down_sampling = Downsampling(patch,num_decoder,sample_channel)

        self.encoder = TransformerLayers(embed_dim, decoder_depth, mlp_ratio, num_heads, dropout)
        self.decoder = TransformerLayers(embed_dim, decoder_depth, mlp_ratio, num_heads, dropout)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim),requires_grad=False)
        
        self.prediction = nn.Linear(embed_dim,embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # positional encoding
        nn.init.uniform_(self.positional_encoding.position_embedding, -.01, .01)
        # mask token
        trunc_normal_(self.mask_token, std=1)

    def encoding(self, long_term_history, mask=True):
        batch_size, num_nodes, _, _ = long_term_history.shape
        patches = self.patch_embedding(long_term_history)# B, N, d, P
        tot= self.num_decoder
        if mask:
            masked_index,unmasked_index = self.mask()
            masked_patch = []
            for i in range(tot):
                trunc_normal_(self.mask_token, mean=0,std=1)
                patches[i][:,:,masked_index[tot-i-1],:] = \
                    self.mask_token.expand(batch_size, num_nodes,
                                           len(masked_index[tot-i-1]), self.embed_dim)
                masked_patch.append(patches[i])
        else:
            masked_index,unmasked_index = None, None
            masked_patch = []
            for i in range(tot):
                masked_patch.append(patches[i])

        maskinput = self.positional_encoding(input=masked_patch)# B, N, d, P  
        hid_states = [0]*tot
        trd_states = [0]*tot
        sea_states = [0]*tot
        ret_states = [0]*tot

        for i in range(0,tot):
            hid_states[i] = self.encoder(maskinput[i])
        
        # trd_states[0]=hid_states[0]
        for i in range(1,tot):
            trd_states[i-1] = self.up_sampling(hid_states[i],i)
        # sea_states[0]=hid_states[tot-1]
        for i in range(0,tot-1):
            sea_states[i+1] = self.down_sampling(hid_states[i],i+1)

        ret_states[0]                 =trd_states[0]+hid_states[0]
        for i in range(1,tot-1):
            ret_states[i] = 0.5*(sea_states[i]+trd_states[i]) + hid_states[i]
        ret_states[tot-1]      =sea_states[tot-1]+hid_states[tot-1]

        return ret_states,unmasked_index,masked_index
    
    def decoding(self,hidden_states):
        # batch_size, num_nodes, _, _ = hidden_states.shape
        # hidden_states = hidden_states.transpose(-1,-2)
        # states_full = self.prediction1(hidden_states).transpose(-1,-2)
        output = [0]*self.num_decoder
        for i in range(0,self.num_decoder):
            output[i] = self.decoder(nn.SiLU()(hidden_states[i]))
        states_full = self.up_summary(output)
        return states_full # torch.Size([12, 500, 12, 12])

    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full, unmasked_token_index, 
                                        masked_token_index,patch):
        """Get reconstructed masked tokens and corresponding ground-truth for subsequent loss computing.

        Args:
            reconstruction_full (torch.Tensor): reconstructed full tokens.
            real_value_full (torch.Tensor): ground truth full tokens.
            unmasked_token_index (list): unmasked token index.
            masked_token_index (list): masked token index.

        Returns:
            torch.Tensor: reconstructed masked tokens.
            torch.Tensor: ground truth masked tokens.
        """
        # get reconstructed masked tokens
        batch_size, num_nodes, _, _ = reconstruction_full.shape
        reconstruction_masked_tokens = reconstruction_full[:, :, :, :]     # B, N, r*P, d
        reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)     # B, r*P*d, N

        label_full = real_value_full.permute(0, 3, 1, 2).unfold(1, patch, patch)[:, :, :, self.selected_feature, :].transpose(1, 2)  # B, N, P, L
        label_masked_tokens = label_full[:, :, :, :].contiguous() # B, N, r*P, d
        label_masked_tokens = label_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)  # B, r*P*d, N
        return reconstruction_masked_tokens, label_masked_tokens

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, epoch: int = None, **kwargs) -> torch.Tensor:
        """feed forward of the TSFormer.
            TSFormer has two modes: the pre-training mode and the forecasting mode,
                                    which are used in the pre-training stage and the forecasting stage, respectively.

        Args:
            history_data (torch.Tensor): very long-term historical time series with shape B, L * P, N, 1.
            r is the masked ratio
        Returns:
            pre-training:
                torch.Tensor: the reconstruction of the masked tokens. Shape [B, L * P * r, N, 1]
                torch.Tensor: the ground truth of the masked tokens. Shape [B, L * P * r, N, 1]
                dict: data for plotting.
            forecasting:
                torch.Tensor: the output of TSFormer of the encoder with shape [B, N, L, 1].
        """

        history_data = history_data.permute(0, 2, 3, 1)     # B, N, 1, L * P
        
        if self.mode == "pre-train":
            hidden_states_unmasked,unmasked_index,masked_index = self.encoding(history_data)
            reconstruction_full = self.decoding(hidden_states_unmasked)
            reconstruction_masked_tokens, label_masked_tokens = \
                self.get_reconstructed_masked_tokens(reconstruction_full, history_data, \
                                                    unmasked_index[-1], masked_index[-1], 1)
            
            return reconstruction_masked_tokens, label_masked_tokens
        else:
            hidden_states_full,_,_ = self.encoding(history_data,mask=False)
            return hidden_states_full