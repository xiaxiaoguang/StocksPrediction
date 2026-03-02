import torch
from torch import nn
from timm.models.vision_transformer import trunc_normal_
from .patch import PatchEmbedding
from .mask import MaskGenerator
from .positional_encoding import PositionalEncoding
from .positional_encoding import SpacePositionalEncoding
from .polymeric import Polymeric
from .transformer_layers import TransformerLayers,TransformerLayers2

def unshuffle(shuffled_tokens):
    dic = {}
    for k, v, in enumerate(shuffled_tokens):
        dic[v] = k
    unshuffle_index = []
    for i in range(len(shuffled_tokens)):
        unshuffle_index.append(dic[i])
    return unshuffle_index

class SATFormer(nn.Module):
    """An efficient unsupervised pre-training model for Time Series based on transFormer blocks. (TSFormer)"""

    def __init__(self, patch ,space_patch, in_channel,
                 embed_dim, num_heads, mlp_ratio,
                 dropout, num_token, feature_num,mask_ratio,
                 num_decoder,decoder_depth, encoder_depth,
                 selected_feature,
                 mode="pre-train"):
        super().__init__()
        assert mode in ["pre-train", "forecasting"], "Error mode."
        # assert time in ["week","month"],"Error version"
        self.patch = patch
        self.patch_size = 1 
        for i in self.patch:
            self.patch_size = self.patch_size * i

        self.num_decoder = num_decoder
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_token = num_token
        self.mask_ratio = mask_ratio
        self.mode = mode
        self.mlp_ratio = mlp_ratio
        self.selected_feature = selected_feature

        # self.encoder_norm = nn.LayerNorm(embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim , patch , dropout=dropout) 
        
        self.space_poly = Polymeric(space_patch)
        
        self.space_mlp = nn.Sequential(nn.Linear(feature_num * 2,embed_dim),nn.ReLU())

        self.space_embed = SpacePositionalEncoding(embed_dim)

        self.encoder=TransformerLayers(embed_dim,encoder_depth,mlp_ratio,num_heads,dropout)
        
        self.patch_embedding = PatchEmbedding(patch , in_channel, embed_dim ,norm_layer=None, num_feats=embed_dim)
        
        self.device =next(self.parameters()).device    

        self.mask = MaskGenerator(num_token ,patch , mask_ratio , self.device)

        self.decoder = nn.ModuleList()
        for i in range(num_decoder):
            self.decoder.append(TransformerLayers2(embed_dim, decoder_depth[i], mlp_ratio, num_heads, dropout))
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        
        self.prediction1 = nn.Sequential(nn.Linear(num_token,num_token),nn.ReLU())
        self.prediction2 = nn.Linear(embed_dim,self.patch_size)

        self.initialize_weights()

    def initialize_weights(self):
        # positional encoding
        nn.init.uniform_(self.positional_encoding.position_embedding, -.01, .01)
        # mask token
        trunc_normal_(self.mask_token, std=1)

    def encoding(self, long_term_history, mask=True):
        """Encoding process of TSFormer: patchify, positional encoding, mask, Transformer layers.

        Args:
            long_term_history (torch.Tensor): Very long-term historical MTS with shape [B, N, 1, P * L],
                                                which is used in the TSFormer.
                                                P is the number of segments (patches).
            mask (bool): True in pre-training stage and False in forecasting stage.

        Returns:
            torch.Tensor: hidden states of unmasked tokens
            list: unmasked token index
            list: masked token index
        """
        
        space_patch = long_term_history.permute(0,3,1,2)

        space_patch = self.space_poly(space_patch)
        space_patch = self.space_mlp(space_patch) 
        
        space_patch = self.space_embed(space_patch)# B,P,N,d
        space_patch = self.encoder(space_patch).permute(0,2,3,1)

        batch_size, num_nodes, _, _ = space_patch.shape
        patches = self.patch_embedding(space_patch)# B, N, d, P

        for i in range(self.num_decoder):
            patches[i]=patches[i].transpose(-1,-2)
        patches = self.positional_encoding(input=patches)# B, N, d, P
        
        if mask:
            masked_index,unmasked_index = self.mask()
            maskinput = []
            masked_patch = []
            trunc_normal_(self.mask_token, mean=patches[0].mean(),std=1)
            for i in range(self.num_decoder):
                masked_patch.append(self.mask_token.expand(batch_size, num_nodes, len(masked_index[i]), self.embed_dim))
            masked_patch = self.positional_encoding(
                input = masked_patch,
                index = masked_index)
            for i in range(self.num_decoder):
                tmp = patches[i][:,:,unmasked_index[self.num_decoder-i-1],:]
                states_full = torch.cat([tmp, masked_patch[self.num_decoder-i-1]], dim=-2)# B, N, P, d 
                maskinput.append(states_full)
        else:
            masked_index,unmasked_index = None, None
            maskinput = []
            for i in range(self.num_decoder):
                maskinput.append(patches[i])

        maskinput[0] = self.decoder[0](tgt=maskinput[0],memory=maskinput[0])  

        ret_states=[maskinput[0][:,:,-1,:]]
        for i in range(1,self.num_decoder):
            maskinput[i] = self.decoder[i](tgt=maskinput[i],memory=maskinput[i-1])
            ret_states.append(maskinput[i][:,:,-1,:])

        return maskinput[self.num_decoder-1],unmasked_index,masked_index,ret_states
    
    def decoding(self,hidden_states):
        """Decoding process of TSFormer: encoder 2 decoder layer, add mask tokens, Transformer layers, predict.

        Args:
            hidden_states_unmasked (torch.Tensor): hidden states of masked tokens [B, N, P*(1-r), d].
            masked_token_index (list): masked token index

        Returns:
            torch.Tensor: reconstructed data
        """
        batch_size, num_nodes, _, _ = hidden_states.shape
        hidden_states = hidden_states.transpose(-1,-2)
        states_full = self.prediction1(hidden_states).transpose(-1,-2)
        states_full = self.prediction2(states_full)
        return states_full # torch.Size([12, 500, 12, 12])

    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full, unmasked_token_index, masked_token_index):
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
        reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_index):, :]     # B, N, r*P, d
        reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)     # B, r*P*d, N
        # breakpoint()# real_value_full B,N,d,P -> B,P,N,d
        real_value_full = self.space_poly(real_value_full.permute(0 , 3 , 1 , 2))
        label_full = real_value_full.unfold(1, self.patch_size, self.patch_size)[:, :, :, self.selected_feature, :].transpose(1, 2)  # B, N, P, L

        label_masked_tokens = label_full[:, :, masked_token_index, :].contiguous() # B, N, r*P, d
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
            hidden_states_unmasked,unmasked_index,masked_index,_ = self.encoding(history_data)

            reconstruction_full = self.decoding(hidden_states_unmasked)

            reconstruction_masked_tokens, label_masked_tokens = \
                self.get_reconstructed_masked_tokens(reconstruction_full, history_data, \
                                                    unmasked_index[0], masked_index[0])
            
            return reconstruction_masked_tokens, label_masked_tokens
        else:
            hidden_states_full2,_,_,hidden_states_full = self.encoding(history_data,mask=False)
            return hidden_states_full