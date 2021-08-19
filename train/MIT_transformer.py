import torch.nn as nn
import torch as tr
import numpy as np
from MIT_layers import *

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class PositionalEncoding(nn.Module):

    def __init__(self, hidden_dim, num_pos=200):
        super(PositionalEncoding, self).__init__()

        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(num_pos, hidden_dim))

    def _get_sinusoid_encoding_table(self, num_pos, hidden_dim):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / hidden_dim) \
                    for hid_j in range(hidden_dim)]
        sinusoid_table = np.array([get_position_angle_vec(pos_idx) for pos_idx in range(num_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return tr.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()



class Encoder(nn.Module):

    def __init__(self, src_dim, emb_dim, pad_ind, model_dim, inner_dim, num_head,
                 k_dim, v_dim, num_layer, dr=0.1, num_pos=200, scale_emb=False):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_dim, emb_dim, padding_idx=pad_ind)
        self.pos_enc = PositionalEncoding(emb_dim, num_pos=num_pos)
        self.dr = nn.Droupout(p=dr)
        self.layer_stack = nn.ModuleList([EncoderLayer(model_dim, inner_dim, num_head, k_dim,
                                                       v_dim, dr=dr) for _ in range(num_layer)])
        self.layer_norm = nn.LayerNorm(model_dim, eps=1e-6)
        self.scale_emb = scale_emb
        self.model_dim = model_dim

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        enc_output = self.src_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.model_dim ** 0.5
        enc_output = self.dr(self.pos_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output

class Decoder(nn.Module):

    def __init__(self, trg_dim, emb_dim, pad_ind, model_dim, inner_dim, num_head,
                 k_dim, v_dim, num_layer, dr=0.1, num_pos=200, scale_emb=False):

        super().__init__()

        self.trg_emb = nn.Embedding(trg_dim, emb_dim, padding_idx=pad_ind)
        self.pos_enc = PositionalEncoding(emb_dim, n_position=num_pos)
        self.dr = nn.Dropout(p=dr)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(model_dim, inner_dim, num_head, k_dim, v_dim, dropout=dr)
            for _ in range(num_layer)])
        self.layer_norm = nn.LayerNorm(model_dim, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = model_dim

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.trg_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dr(self.pos_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output

class Transformer(nn.Module):

    def __init__(self, window, src_dim, trg_dim, src_pad_ind, trg_pad_ind,
                 emb_dim=512, model_dim=512, inner_dim=2048, num_layer=6, num_head=8,
                 k_dim=64, v_dim=64, dr=0.1, num_pos=200, trg_emb_prj_weight_sharing=True,
                 emb_src_trg_weight_sharing=True,scale_emb_or_prj='prj'):
        super(Transformer, self).__init__()
        '''
             n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj'
        '''
        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.model_dim = model_dim

        self.encoder = Encoder(
            src_dim, emb_dim, src_pad_ind, model_dim, inner_dim, num_head,
            k_dim, v_dim, num_layer, dr=dr, num_pos=num_pos, scale_emb=scale_emb
        )

        self.decoder = Decoder(
            trg_dim, emb_dim, trg_pad_ind, model_dim, inner_dim, num_head,
            k_dim, v_dim, num_layer, dr=dr, num_pos=num_pos, scale_emb=scale_emb
        )

        self.trg_word_prj = nn.Linear(model_dim, trg_dim, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert model_dim == emb_dim, \
            'To facilitate the residual connections, \
             the dimensions of all module outputs shall be the same.'

        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_emb.weight

        if emb_src_trg_weight_sharing:
            self.encoder.src_emb.weight = self.decoder.trg_emb.weight

    def forward(self, src_seq, trg_seq):

        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output)
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5

        return seq_logit.view(-1, seq_logit.size(2))