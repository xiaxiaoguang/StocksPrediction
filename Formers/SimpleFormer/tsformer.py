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

class SimpleFormer(nn.Module):
    """An efficient unsupervised pre-training model for Time Series based on transFormer blocks. (TSFormer)"""

    def __init__(self, patch_size,node_fusion, 
                 embed_dim, embed_dim2,
                 num_heads, mlp_ratio,
                 dropout, num_token,num_nodes,
                 encoder1_depth, encoder2_depth, decoder_depth,
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
        self.mode = mode
        self.mlp_ratio = mlp_ratio
        self.selected_feature = selected_feature

        self.enc_2_enc_norm_ti = nn.BatchNorm1d(num_nodes*node_fusion)
        self.enc_2_enc_norm_sp = nn.BatchNorm1d(num_token*patch_size)

        self.encoder_norm2_ti = nn.InstanceNorm1d(embed_dim2)
        self.encoder_norm2_sp = nn.LayerNorm(embed_dim2)

        self.decoder1_norm = nn.LayerNorm(embed_dim2)
        self.decoder2_norm = nn.InstanceNorm2d(embed_dim2)

        self.patch_embedding = STPatchEmbedding(patch_size, node_fusion,
                                                 embed_dim ,norm_layer=None, num_feats=num_feats)
        
        self.positional_encoding_t = OldPositionalEncoding(embed_dim2, dropout=dropout)
        self.positional_encoding_s = OldPositionalEncoding(embed_dim2, dropout=dropout)

        self.device = next(self.parameters()).device

        # self.mask = MaskGenerator(num_token, mask_ratio , self.device)
        self.enc_2_enc_emb_ti1 = nn.Linear(num_token*embed_dim,num_token*embed_dim//4)
        self.enc_2_enc_emb_ti2 = nn.Linear(num_token*embed_dim//4,embed_dim2)

        self.enc_2_enc_emb_sp1 = nn.Linear(num_nodes*embed_dim,num_nodes*embed_dim//4)
        self.enc_2_enc_emb_sp2 = nn.Linear(num_nodes*embed_dim//4,embed_dim2)

        self.tiencoder2 = TransformerLayers(embed_dim2, encoder1_depth, mlp_ratio,num_heads, dropout)
        self.spencoder2 = TransformerLayers(embed_dim2, encoder2_depth, mlp_ratio,num_heads, dropout)

        # self.enc_2_dec_emb_ti = nn.Linear(embed_dim2, embed_dim2, bias=True)
        # self.enc_2_dec_emb_sp = nn.Linear(embed_dim2, embed_dim2, bias=True)

        # self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        # self.decoder  = TransformerLayers2(embed_dim2, decoder_depth,   mlp_ratio, num_heads, dropout)
        # self.decoder2 = TransformerLayers(embed_dim2 ,decoder_depth,    mlp_ratio, num_heads, dropout)
        # self.output_layer = nn.Linear(embed_dim2, pred_len)
        self.out = nn.Sequential(
            nn.Linear(embed_dim2, embed_dim2 // 2),
            nn.ReLU(),
            nn.Linear(embed_dim2 // 2, pred_len),
        )
        self.saliency_proj1 = nn.Linear(embed_dim2,embed_dim2)
        self.saliency_proj2 = nn.Linear(embed_dim2,embed_dim2)
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
        batch_size, num_nodes, _, tim_len = long_term_history.shape# patchify and embed input

        ti_patches,sp_patches = self.patch_embedding(long_term_history)     # B, N, d, P
        ti_patches = ti_patches.transpose(-1, -2) # B, N, P, d
        sp_patches = sp_patches.transpose(-1 ,-2)

        hidden_states_ti = self.enc_2_enc_emb_ti1(ti_patches.reshape(batch_size, num_nodes, -1))
        hidden_states_ti = self.enc_2_enc_norm_ti(hidden_states_ti)
        hidden_states_ti = nn.ReLU()(hidden_states_ti)
        hidden_states_ti = self.enc_2_enc_emb_ti2(hidden_states_ti)

        # hidden_states_ti = self.positional_encoding_t(hidden_states_ti)
        # hidden_states_ti = self.tiencoder2(hidden_states_ti)
        # hidden_states_ti = self.encoder_norm2_ti(hidden_states_ti)

        hidden_states_sp = self.enc_2_enc_emb_sp1(sp_patches.reshape(batch_size, tim_len, -1))
        hidden_states_sp = self.enc_2_enc_norm_sp(hidden_states_sp)
        hidden_states_sp = nn.ReLU()(hidden_states_sp)
        hidden_states_sp = self.enc_2_enc_emb_sp2(hidden_states_sp)

        # hidden_states_sp = self.positional_encoding_s(hidden_states_sp)
        # hidden_states_sp = self.spencoder2(hidden_states_sp)
        # hidden_states_sp = self.encoder_norm2_sp(hidden_states_sp)

        return hidden_states_ti,hidden_states_sp
    # , unmasked_token_index, masked_token_index
    
    def decoding(self, hidden_states_ti,hidden_states_sp):
        """Decoding process of TSFormer: encoder 2 decoder layer, add mask tokens, Transformer layers, predict.

        Args:
            hidden_states_unmasked (torch.Tensor): hidden states of masked tokens [B, N, P*(1-r), d].
            masked_token_index (list): masked token index

        Returns:
            torch.Tensor: reconstructed data
        """
        # breakpoint()
        # B,N = hidden_states_ti.shape[0],hidden_states_ti.shape[1]
        # L = hidden_states_sp.shape[1]
        # hidden_states_ti = hidden_states_ti.unsqueeze(2).repeat(1,1,L,1)
        # hidden_states_sp = hidden_states_sp.unsqueeze(1).repeat(1,N,1,1)
        # pred = torch.cat([hidden_states_ti,hidden_states_sp],dim=-1)
        pred = self.out(hidden_states_ti)
        # pred2 = self.out2(hidden_states_sp)
        return pred
        # pred = self.decoder(memory=hidden_states_sp,tgt=hidden_states_ti)
        # pred = self.decoder1_norm(pred)

        # hidden_states_ti = self.decoder2(hidden_states_ti)
        # hidden_states_ti = self.decoder2_norm(hidden_states_ti)

        # pred = hidden_states_ti + pred
        # pred = self.output_layer(pred)
        # return pred # torch.Size([12, 500, 12, 12])

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
            hidden_states_ti,hidden_states_sp = self.encoding(history_data)
            reconstruction_full = self.decoding(hidden_states_ti,hidden_states_sp)

            # if batch_seen is not None and batch_seen % 200 == 0 and batch_seen != 0:
            #     with open("test-1.out",'a') as f:
            #         print("recons:", reconstruction_full[0,:5,:],file=f)
            #         print("future:",future_data[0,:5,:,self.selected_feature],file=f)
            #         print("linear weight/grad:",self.enc_2_enc_emb_sp2.weight[0,:10],self.enc_2_enc_emb_sp2.weight.grad[0,:10],file=f)
                    # print("linear2weight/grad:",self.enc_2_dec_emb_sp.weight[0,:10],self.enc_2_dec_emb_sp.weight.grad[0,:10],file=f)
                    # print("attention weight/grad:",self.decoder.transformer_decoder.layers.multihead_attn.q_proj_weight[0,:10],
                    #       self.decoder.transformer_decoder.layers.multihead_attn.q_proj_weight.grad[0,:10],file=f)

            return reconstruction_full, future_data[:,:,:,self.selected_feature],\
                self.saliency_proj1(hidden_states_ti), self.saliency_proj2(hidden_states_sp)
        else:

            hidden_states_ti,hidden_states_sp = self.encoding(history_data,mask=False)
            reconstruction_full = self.decoding(hidden_states_ti,hidden_states_sp)

            return hidden_states_ti,reconstruction_full