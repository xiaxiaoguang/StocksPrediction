import torch
from torch import nn
from timm.models.vision_transformer import trunc_normal_

from .patch import STPatchEmbedding
from .positional_encoding import OldPositionalEncoding
from .transformer_layers import TransformerLayers,TransformerLayers2

def unshuffle(shuffled_tokens):
    dic = {}
    for k, v, in enumerate(shuffled_tokens):
        dic[v] = k
    unshuffle_index = []
    for i in range(len(shuffled_tokens)):
        unshuffle_index.append(dic[i])
    return unshuffle_index

class CAOFormer2(nn.Module):
    """An efficient unsupervised pre-training model for Time Series based on transFormer blocks. (TSFormer)"""

    def __init__(self, patch_size,node_fusion, 
                 embed_dim, embed_dim2,
                 num_heads, mlp_ratio,
                 dropout, num_token,num_nodes,
                 encoder_depth, decoder_depth,
                 selected_feature, pred_len=12,num_feats=45,
                 mode="pre-train"):
        super().__init__()
        # assert mode in ["pre-train", "forecasting"], "Error mode."
        # assert time in ["week","month"],"Error version"
        self.patch_size = patch_size

        self.embed_dim = embed_dim
        self.embed_dim2 = embed_dim2

        self.num_heads = num_heads
        self.num_token = num_token

        self.encoder_depth = encoder_depth
        self.mode = mode
        self.mlp_ratio = mlp_ratio
        self.selected_feature = selected_feature

        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.encoder_norm2 = nn.LayerNorm(embed_dim2)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm2 = nn.LayerNorm(embed_dim2)

        self.patch_embedding = STPatchEmbedding(patch_size, node_fusion,
                                                 embed_dim ,norm_layer=None, num_feats=num_feats)
        self.positional_encoding_t = OldPositionalEncoding(embed_dim, dropout=dropout)
        self.positional_encoding_s = OldPositionalEncoding(embed_dim, dropout=dropout)
        self.device = next(self.parameters()).device
        # self.mask = MaskGenerator(num_token, mask_ratio , self.device)
        self.enc_2_enc_emb = nn.Linear(num_token*embed_dim,embed_dim2)
        self.dec_2_dec_emb = nn.Linear(num_nodes*embed_dim,embed_dim2)        

        self.encoder1  = TransformerLayers(embed_dim, encoder_depth[0], mlp_ratio, num_heads, dropout)
        self.encoder2 = TransformerLayers(embed_dim2,encoder_depth[1], mlp_ratio,num_heads, dropout)

        self.enc_2_dec_emb = nn.Linear(embed_dim2, embed_dim2, bias=True)
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        self.decoder1  = TransformerLayers(embed_dim, decoder_depth[0], mlp_ratio, num_heads, dropout)
        self.decoder2  = TransformerLayers2(embed_dim2, decoder_depth[1], mlp_ratio, num_heads, dropout)

        self.output_layer = nn.Linear(embed_dim2, pred_len)
        
        self.initialize_weights()

    def initialize_weights(self):
        # positional encoding
        nn.init.uniform_(self.positional_encoding_s.position_embedding, -.02, .02)
        nn.init.uniform_(self.positional_encoding_t.position_embedding, -.02, .02)
        # mask token
        # trunc_normal_(self.mask_token, std=.02)

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
        batch_size, num_nodes, _, _ = long_term_history.shape# patchify and embed input
        patches,sp_states = self.patch_embedding(long_term_history)     # B, N, d, P
        patches = patches.transpose(-1, -2) # B, N, P, d

        patches = self.positional_encoding_t(patches)

        hidden_states_unmasked = self.encoder1(patches)
        hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_nodes, -1)
        hidden_states_unmasked = self.enc_2_enc_emb(hidden_states_unmasked)

        hidden_states_unmasked = self.encoder2(hidden_states_unmasked)
        hidden_states_unmasked = self.encoder_norm2(hidden_states_unmasked)

        return hidden_states_unmasked,sp_states
    # , unmasked_token_index, masked_token_index
    
    def decoding(self, hidden_states_unmasked,full_states):
        """Decoding process of TSFormer: encoder 2 decoder layer, add mask tokens, Transformer layers, predict.

        Args:
            hidden_states_unmasked (torch.Tensor): hidden states of masked tokens [B, N, P*(1-r), d].
            masked_token_index (list): masked token index

        Returns:
            torch.Tensor: reconstructed data
        """
        hidden_states = nn.GELU()(self.enc_2_dec_emb(hidden_states_unmasked))

        batch_size,time_len,_,num_nodes = full_states.shape
        full_states = self.positional_encoding_s(full_states.reshape(batch_size,time_len,num_nodes,-1))
        
        full_states = self.decoder1(full_states)
        full_states = self.decoder_norm(full_states).reshape(batch_size,time_len,-1)
        full_states = self.dec_2_dec_emb(full_states)

        hidden_states_full = self.decoder2(memory=full_states,tgt=hidden_states)
        hidden_states_full = self.decoder_norm2(hidden_states_full)
        
        reconstruction_full = self.output_layer(hidden_states_full)
        return reconstruction_full,full_states # torch.Size([12, 500, 12, 12])

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
            future_data = future_data.permute(0, 2 , 1 , 3)
            hidden_states_alongs,full_states = self.encoding(history_data)
            reconstruction_full,hidden_states_alongt = self.decoding(hidden_states_alongs,full_states)

            return reconstruction_full, future_data[:,:,:,self.selected_feature],hidden_states_alongs,hidden_states_alongt
        else:
            hidden_states_full,full_states = self.encoding(history_data,mask=False)
            reconstruction_full,hidden_states_alongt = self.decoding(hidden_states_full,full_states)

            return hidden_states_full,reconstruction_full