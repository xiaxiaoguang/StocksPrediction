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

class SMTSF(nn.Module):
    """An efficient unsupervised pre-training model for Time Series based on transFormer blocks. (TSFormer)"""

    def __init__(self, patch_sizeA ,leverage ,vision, in_channel,
                 embed_dim, num_heads, mlp_ratio,
                 dropout, num_token, mask_ratio,
                 week_encoder_depth, month_decoder_depth, space_decoder_depth,final_decoder_depth,
                 selected_feature,
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
        self.mode = mode
        self.mlp_ratio = mlp_ratio
        self.selected_feature = selected_feature

        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)

        self.patch_embedding = PatchEmbedding(patch_sizeA ,leverage ,vision, in_channel, 
                                              num_token, embed_dim ,norm_layer=None)
        self.positional_encoding = PositionalEncoding(embed_dim,patch_sizeA , dropout=dropout)

        self.device = next(self.parameters()).device

        # self.time_mask = MaskGenerator(num_token ,leverage , mask_ratio , self.device)
        # self.space_mask = MaskGenerator(num_nodes , None, mask_ratio , self.device)

        self.week_encoder = TransformerLayers(embed_dim, week_encoder_depth, mlp_ratio, num_heads, dropout)
        self.month_decoder = TransformerLayers2(embed_dim,month_decoder_depth,mlp_ratio,num_heads,dropout)
        self.space_decoder = TransformerLayers2(embed_dim,space_decoder_depth,mlp_ratio,num_heads,dropout)
        self.final_decoder = TransformerLayers(embed_dim,final_decoder_depth,mlp_ratio,num_heads,dropout)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        
        self.before_decoder1 = nn.Linear(embed_dim, embed_dim)
        self.before_decoder2 = nn.Linear(embed_dim, embed_dim)
        self.before_activate = nn.ReLU()
        self.decoder =         EasyDecoder(embed_dim , embed_dim*2)
        self.output_layer2 = nn.Linear(embed_dim, self.patch_size)
        # self.output_layer1 = nn.Linear(embed_dim, 1)
        self.initialize_weights()

    def initialize_weights(self):
        # positional encoding
        nn.init.uniform_(self.positional_encoding.position_embedding, -.02, .02)
        nn.init.uniform_(self.positional_encoding.space_embedding, -.03, .03)
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
        patch1,patch2,patch3 = self.patch_embedding(long_term_history)     # B, N, d, P
        patch1 = patch1.transpose(-1, -2) # B, N, P, d
        patch2 = patch2.transpose(-1, -2)
        patch3 = patch3.transpose(-1, -2) # B,P,N,d
        # positional embedding
        patch1,patch2,patch3 = self.positional_encoding(input1=patch1,input2=patch2,input3=patch3)
        # mask
        if mask:
            num_tokens = patch2.shape[-2]
            leverage = patch1.shape[-2] // patch2.shape[-2]
            num_nodes = patch3.shape[-2]

            self.time_mask = MaskGenerator(num_tokens ,leverage , self.mask_ratio , self.device)
            self.space_mask = MaskGenerator(num_nodes,  None     ,self.mask_ratio,  self.device)

            unmasked_index1, masked_index1, unmasked_index2,masked_index2 = self.time_mask()
            unmasked_index3, masked_index3 = self.space_mask()
            
            mask_input1 = patch1[:, :, unmasked_index1, :]
            mask_input2 = patch2[:, :, unmasked_index2, :]
            mask_input3 = patch3[:, :, unmasked_index3, :]
        else:
            unmasked_index1, unmasked_index2, masked_index1 ,masked_index2,unmasked_index3, masked_index3\
                  = None, None ,None ,None, None ,None
            mask_input1 = patch1
            mask_input2 = patch2
            mask_input3 = patch3
        # encoding
        hidden_states_unmasked1 = self.week_encoder(mask_input1)
        # hidden_states_unmasked2 = self.week_encoder(mask_input3)
        month_states = self.month_decoder(tgt=mask_input2,memory=hidden_states_unmasked1)
        if mask:
            month_to_space = self.month_decoder(tgt=patch2   ,memory=hidden_states_unmasked1).transpose(1,2)
        else :
            month_to_space = month_states.transpose(1,2)
            
        space_states = self.space_decoder(tgt=mask_input3,memory=month_to_space)
        # hidden_states_unmasked = self.enc_2_dec_emb(hidden_states_unmasked)
        return month_states , space_states, \
             unmasked_index2 ,unmasked_index3 ,\
                 masked_index2 ,masked_index3
    
    def decoding(self, hidden_states1, masked_index1 ,hidden_states2,masked_index2):
        """Decoding process of TSFormer: encoder 2 decoder layer, add mask tokens, Transformer layers, predict.

        Args:
            hidden_states_unmasked (torch.Tensor): hidden states of masked tokens [B, N, P*(1-r), d].
            masked_token_index (list): masked token index

        Returns:
            torch.Tensor: reconstructed data
        """
        batch_size, num_nodes, _, _ = hidden_states2.shape
        num_patches = hidden_states1.shape[1]

        states_masked2 = self.positional_encoding(
            input2=self.mask_token.expand(batch_size, num_nodes, len(masked_index2), hidden_states2.shape[-1]),
            index=masked_index2
            )
        states_full2 = torch.cat([hidden_states2, states_masked2], dim=-2) # B, N, P, d
        month_states = self.before_activate(self.before_decoder2(states_full2))
        # month_states = self.decoder(month_states)
        # month_states = self.decoder_norm(month_states)
        month_states = self.final_decoder(month_states)         
        month_states = self.output_layer2(month_states.view(batch_size, num_nodes, -1, self.embed_dim))
        
        states_masked1 = self.positional_encoding(
            input3=self.mask_token.expand(batch_size, num_patches, len(masked_index1) , hidden_states1.shape[-1]),
            index=masked_index1
        )
        states_full1 = torch.cat([hidden_states1,states_masked1], dim=-2) # B, P , N, d
        num_patches3 = states_full1.shape[-2]
        space_states = self.before_activate(self.before_decoder1(states_full1))
        # space_states = self.decoder(space_states)
        # space_states = self.decoder_norm(space_states)
        space_states = self.final_decoder(space_states)
        space_states = self.output_layer2(space_states.reshape(batch_size, num_patches3, -1, self.embed_dim))
        
        return month_states,space_states 

    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full, unmasked_token_index, masked_token_index, patchsize,transpose=False):
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
        if transpose is True:
            reconstruction_full = reconstruction_full.transpose(1,2)
            reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_index):, :]
            reconstruction_masked_tokens = reconstruction_masked_tokens.transpose(2,3).view(batch_size, -1, len(masked_token_index)).transpose(1, 2)     # B, r*P*d, N
        else :
            reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_index):, :]     # B, N, r*P, d
            reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)     # B, r*P*d, N
        
        if transpose is True:
            num_nodes = reconstruction_masked_tokens.shape[1]
            label_full = real_value_full.permute(0, 1, 3, 2)[:, :, :, self.selected_feature].transpose(1, 2).unsqueeze(-1)
            label_masked_tokens = label_full[:, :, masked_token_index, :].contiguous() # B, N, r*P, d
            label_masked_tokens = label_masked_tokens.view(batch_size, -1,len(masked_token_index)).transpose(1, 2)  # B, r*P*d, N
        else :
            label_full = real_value_full.permute(0, 3, 1, 2).unfold(1, patchsize, patchsize)[:, :, :, self.selected_feature, :].transpose(1, 2)
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
            hidden_states_unmasked2,hidden_states_unmasked3,\
                 unmasked_index2 ,unmasked_index3,\
                 masked_index2 ,masked_index3 = self.encoding(history_data)
            
            month_reconstruction,space_reconstruction = self.decoding(hidden_states_unmasked3,masked_index3,\
                                                hidden_states_unmasked2,masked_index2)
            #torch.Size([1, 500, 50, 20]),torch.Size([1, 200, 500, 5])

            month_masked_tokens, month_label_masked_tokens = \
                self.get_reconstructed_masked_tokens(month_reconstruction, history_data, \
                                                    unmasked_index2, masked_index2,self.patch_size)

            space_masked_tokens, space_label_masked_tokens = \
                self.get_reconstructed_masked_tokens(space_reconstruction, history_data, \
                                                    unmasked_index3, masked_index3,self.patch_sizeA,transpose=True)        
            # 变成利用月信息和空间信息还原周信息了 太抽象了，我是傻逼 兄弟，你是对的
            return month_masked_tokens, month_label_masked_tokens, space_masked_tokens, space_label_masked_tokens
        else:
            month_state,space_state,_,_,_,_ = self.encoding(history_data,mask=False)
            return month_state,space_state