import torch
from torch import nn
from timm.models.vision_transformer import trunc_normal_

from .patch import PatchEmbedding
from .mask import MaskGenerator
from .positional_encoding import PositionalEncoding
from .transformer_layers import EasyDecoder,TransformerLayers,TransformerLayers2

def unshuffle(shuffled_tokens):
    dic = {}
    for k, v, in enumerate(shuffled_tokens):
        dic[v] = k
    unshuffle_index = []
    for i in range(len(shuffled_tokens)):
        unshuffle_index.append(dic[i])
    return unshuffle_index

class MTSFormer2(nn.Module):
    """An efficient unsupervised pre-training model for Time Series based on transFormer blocks. (TSFormer)"""

    def __init__(self, patch_sizeA ,leverage , in_channel,
                 embed_dim, num_heads, mlp_ratio,
                 dropout, num_token, mask_ratio,num_feats,
                 encoder_depth, decoder_depth, selected_feature,
                 mode="pre-train"):
        super().__init__()
        assert mode in ["pre-train", "forecasting"], "Error mode."
        # assert time in ["week","month"],"Error version"
        self.patch_sizeA = patch_sizeA
        self.patch_size = patch_sizeA * leverage

        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_token = num_token
        self.mask_ratio = mask_ratio
        self.encoder_depth = encoder_depth
        self.mode = mode
        self.mlp_ratio = mlp_ratio
        self.selected_feature = selected_feature

        self.encoder1_norm = nn.LayerNorm(embed_dim)
        self.encoder2_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm  = nn.LayerNorm(embed_dim)

        self.patch_embedding = PatchEmbedding(patch_sizeA ,leverage , in_channel, embed_dim ,
                                              norm_layer=nn.InstanceNorm2d(embed_dim//2),
                                              num_feats=num_feats)

        self.positional_encoding = PositionalEncoding(embed_dim,patch_sizeA , dropout=dropout)

        self.device = next(self.parameters()).device

        self.mask = MaskGenerator(num_token ,leverage , mask_ratio , self.device)
        
        self.encoder1 = TransformerLayers(embed_dim, encoder_depth[0], mlp_ratio, num_heads, dropout)
        self.encoder2 = TransformerLayers(embed_dim, encoder_depth[1],mlp_ratio , num_heads, dropout)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        
        self.decoder = TransformerLayers2(embed_dim, decoder_depth ,mlp_ratio ,num_heads,dropout)

        self.output_layer2 = nn.Linear(embed_dim, self.patch_sizeA)
        
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
        patch1,patch2 = self.patch_embedding(long_term_history)     # B, N, d, P
        patch1 = patch1.transpose(-1, -2)         # B, N, P, d
        patch2 = patch2.transpose(-1, -2)         # positional embedding
        patch1,patch2 = self.positional_encoding(input1=patch1,input2=patch2) # mask

        if mask:
            unmasked_index1, masked_index1,unmasked_index2,masked_index_2 = self.mask()
            mask_input1 = patch1[:, :, unmasked_index1, :]
            mask_input2 = patch2[:, :, unmasked_index2, :]
            # mask_input2[:, :, masked_index_2, :] = self.positional_encoding(
            #     input2=self.mask_token.expand(batch_size, num_nodes, len(masked_index_2), mask_input2.shape[-1]),
            #     index=masked_index_2
            # )
        else:
            unmasked_index1, unmasked_index2,masked_index1,masked_index_2 = None, None ,None ,None
            mask_input1 = patch1
            mask_input2 = patch2

        # encoding
        hidden_states_unmasked1 = self.encoder1(mask_input1)
        hidden_states_unmasked1 = self.encoder1_norm(hidden_states_unmasked1)# hidden_states_full torch.Size([12, 500, 12, 512])
        
        hidden_states_unmasked2 = self.encoder2(mask_input2)
        hidden_states_unmasked2 = self.encoder2_norm(hidden_states_unmasked2)# hidden_states_full torch.Size([12, 500, 12, 512])

        # hidden_states_unmasked = self.encoder2(tgt=mask_input2,memory=hidden_states_unmasked1)
        # hidden_states_unmasked = self.encoder2_norm(hidden_states_unmasked)# hidden_states_full torch.Size([12, 500, 12, 512])
        # hidden_states_unmasked = self.enc_2_dec_emb(hidden_states_unmasked)
        return hidden_states_unmasked1,hidden_states_unmasked2, unmasked_index1, unmasked_index2 ,masked_index1 , masked_index_2
    
    def decoding(self, hidden_states1, masked_index1 ,hidden_states2 , masked_index2):
        """Decoding process of TSFormer: encoder 2 decoder layer, add mask tokens, Transformer layers, predict.

        Args:
            hidden_states_unmasked (torch.Tensor): hidden states of masked tokens [B, N, P*(1-r), d].
            masked_token_index (list): masked token index

        Returns:
            torch.Tensor: reconstructed data
        """
        batch_size, num_nodes, _, _ = hidden_states1.shape
        
        states_masked1 = self.positional_encoding(
            input1=self.mask_token.expand(batch_size, num_nodes, len(masked_index1), hidden_states1.shape[-1]),
            index=masked_index1
            )
        states_full1 = torch.cat([hidden_states1, states_masked1], dim=-2)   # B, N, P, d

        states_masked2 = self.positional_encoding(
            input2=self.mask_token.expand(batch_size, num_nodes, len(masked_index2), hidden_states2.shape[-1]),
            index=masked_index2
            )
        states_full2 = torch.cat([hidden_states2, states_masked2], dim=-2)   # B, N, P, d
        # states_full2 = self.decoder(states_full2)
        states_full1 = self.decoder(tgt=states_full1,memory=states_full2)
        states_full1 = self.decoder_norm(states_full1)# hidden_states_full torch.Size([12, 500, 12, 512])
        states_full1 = self.output_layer2(states_full1.view(batch_size, num_nodes, -1, self.embed_dim))
        return states_full1 # torch.Size([12, 500, 12, 12])

    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full, unmasked_token_index, masked_token_index,patchsize):
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

        label_full = real_value_full.permute(0, 3, 1, 2).unfold(1, patchsize, patchsize)[:, :, :, self.selected_feature, :].transpose(1, 2)  # B, N, P, L
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
            hidden_states_unmasked1,hidden_states_unmasked2,\
                unmasked_index1, unmasked_index2 ,\
                masked_index1, masked_index_2 = self.encoding(history_data)
            
            reconstruction_full = self.decoding(hidden_states_unmasked1,masked_index1,\
                                                hidden_states_unmasked2, masked_index_2)
            
            reconstruction_masked_tokens, label_masked_tokens = \
                self.get_reconstructed_masked_tokens(reconstruction_full, history_data, \
                                                    unmasked_index1, masked_index1,\
                                                    self.patch_sizeA)
            return reconstruction_masked_tokens, label_masked_tokens
        else:
            _,hidden_states_full,_,_,_,_ = self.encoding(history_data,mask=False)
            return hidden_states_full