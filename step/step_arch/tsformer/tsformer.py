import torch
from torch import nn
from timm.models.vision_transformer import trunc_normal_

from .patch import PatchEmbedding,OldPatchEmbedding
from .mask import MaskGenerator
from .positional_encoding import OldPositionalEncoding
from .transformer_layers import TransformerLayers,TransformerEncoder2,TransformerEncoderLayer2,TransformerDecoderLayer,TransformerDecoder


def unshuffle(shuffled_tokens):
    dic = {}
    for k, v, in enumerate(shuffled_tokens):
        dic[v] = k
    unshuffle_index = []
    for i in range(len(shuffled_tokens)):
        unshuffle_index.append(dic[i])
    return unshuffle_index

class TSFormer(nn.Module):
    """An efficient unsupervised pre-training model for Time Series based on transFormer blocks. (TSFormer)"""

    def __init__(self, patch_size, in_channel, embed_dim, num_heads, mlp_ratio,
                 dropout, num_token, mask_ratio,
                 encoder_depth, decoder_depth, selected_feature, 
                 mode="pre-train",num_feats=1):
        super().__init__()
        # assert mode in ["pre-train", "forecasting"], "Error mode."
        # assert time in ["week","month"],"Error version"
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.num_token = num_token
        self.mask_ratio = mask_ratio
        self.encoder_depth = encoder_depth
        self.mode = mode
        self.mlp_ratio = mlp_ratio
        self.selected_feature = selected_feature
        # self.return_intermediate = return_intermediate
        # norm layers
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        # encoder specifics
        # # patchify & embedding , patch 可以认为是融合一个时间段的信息来提取高维特征
        # if time=="Old":
        #     self.patch_embedding = OldPatchEmbedding(patch_size, in_channel, embed_dim,norm_layer=None)
        # else :
        self.patch_embedding = OldPatchEmbedding(patch_size, in_channel, embed_dim,num_feats=num_feats)
        # # positional encoding
        # if time == "Old":
        self.positional_encoding = OldPositionalEncoding(embed_dim, dropout=dropout)
        # else :
        #     self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout , maxPatch=num_token)
        # # masking
        self.device = next(self.parameters()).device
        self.mask = MaskGenerator(num_token, mask_ratio , self.device)
        # num_token应该是len/mask_ratio!
        # encoder
        # if time=="Old":
        self.encoder = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)
        # else :
        #     # encoder_layer = TransformerEncoderLayer(d_model=embed_dim,nhead=num_heads)
        #     self.encoder = TransformerEncoder2(TransformerEncoderLayer2(d_model=embed_dim,nhead=num_heads),encoder_depth,return_intermediate=self.return_intermediate)
        # decoder specifics
        # transform layer
        self.enc_2_dec_emb = nn.Linear(embed_dim, embed_dim, bias=True)
        # # mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        # # decoder
        self.decoder = TransformerLayers(embed_dim, decoder_depth, mlp_ratio, num_heads, dropout)
        # else : 
        #     # decoder_layer = TransformerDecoderLayer(d_model = embed_dim , nhead = num_heads)
        #     self.decoder = TransformerDecoder(TransformerDecoderLayer(d_model = embed_dim , nhead = num_heads),decoder_depth)
        # # prediction (reconstruction) layer
        self.output_layer = nn.Linear(embed_dim, patch_size)
        self.initialize_weights()

    def initialize_weights(self):
        # positional encoding
        nn.init.uniform_(self.positional_encoding.position_embedding, -.02, .02)
        # mask token
        trunc_normal_(self.mask_token, std=.02)

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
        batch_size, num_nodes, _, _ = long_term_history.shape
        # patchify and embed input
        patches = self.patch_embedding(long_term_history)     # B, N, d, P
        patches = patches.transpose(-1, -2)         # B, N, P, d
        # positional embedding
        patches = self.positional_encoding(patches,index = torch.arange(0,self.num_token,device=self.device,dtype=torch.long))
        # mask
        if mask:
            unmasked_token_index, masked_token_index = self.mask()
            mask_input = patches[:, :, unmasked_token_index, :]
        else:
            unmasked_token_index, masked_token_index = None, None
            mask_input = patches
            
        # encoding
        hidden_states_unmasked = self.encoder(mask_input)
        hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_nodes, -1, self.embed_dim)

        return hidden_states_unmasked, unmasked_token_index, masked_token_index
    
    def newencoding(self,long_term_history):
        batch_size, num_nodes, _, _ = long_term_history.shape
        patches = self.patch_embedding(long_term_history)     # B, N, d, P
        patches = patches.transpose(-1, -2)                   # B, N, P, d
        patches = self.positional_encoding(patches,index = torch.arange(0,self.num_token,device=self.device,dtype=torch.long))
        mask_input = patches 
        hidden_states_unmasked = self.encoder(mask_input)
        # with open("test1.out",'a') as f:
        #     print(long_term_history[0][0][0],mask_input[0][0][0],hidden_states_unmasked[0][0][0],file=f)
        # breakpoint()               
        hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_nodes, -1, self.embed_dim)
        result = self.enc_2_dec_emb(hidden_states_unmasked)
        return result
    
    def decoding(self, hidden_states_unmasked, masked_token_index):
        """Decoding process of TSFormer: encoder 2 decoder layer, add mask tokens, Transformer layers, predict.

        Args:
            hidden_states_unmasked (torch.Tensor): hidden states of masked tokens [B, N, P*(1-r), d].
            masked_token_index (list): masked token index

        Returns:
            torch.Tensor: reconstructed data
        """
        batch_size, num_nodes, _, _ = hidden_states_unmasked.shape

        hidden_states_unmasked = self.enc_2_dec_emb(hidden_states_unmasked)

        # add mask tokens
        hidden_states_masked = self.positional_encoding(
            self.mask_token.expand(batch_size, num_nodes, len(masked_token_index), hidden_states_unmasked.shape[-1]),
            index=masked_token_index
            )
        hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)   # B, N, P, d
        # decoding
        # hidden_states_full = self.mlp()
        hidden_states_full = self.decoder(hidden_states_full)
        hidden_states_full = self.decoder_norm(hidden_states_full) 
        # hidden_states_full torch.Size([12, 500, 12, 512])
        # prediction (reconstruction)
        reconstruction_full = self.output_layer(hidden_states_full.view(batch_size, num_nodes, -1, self.embed_dim))

        return reconstruction_full # torch.Size([12, 500, 12, 12])

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

        label_full = real_value_full.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)[:, :, :, self.selected_feature, :].transpose(1, 2)  # B, N, P, L
        label_masked_tokens = label_full[:, :, masked_token_index, :].contiguous() # B, N, r*P, d
        label_masked_tokens = label_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)  # B, r*P*d, N
        return reconstruction_masked_tokens, label_masked_tokens

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, epoch: int = None, **kwargs) -> torch.Tensor:
        """feed forward of the TSFormer.
            TSFormer has two modes: the pre-training mode and the forecasting mode,
                                    which are used in the pre-training stage and the forecasting stage, respectively.

        Args:
            history_data (torch.Tensor): very long-term historical time series with shape B, L * P, N, 1.

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
            hidden_states_unmasked, unmasked_token_index, masked_token_index = self.encoding(history_data)
            reconstruction_full = self.decoding(hidden_states_unmasked, masked_token_index)
            reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(reconstruction_full, history_data, \
                                                                                                    unmasked_token_index, masked_token_index)
            return reconstruction_masked_tokens, label_masked_tokens
        else:
            hidden_states_full,_,_ = self.encoding(history_data,mask=False)
            return hidden_states_full