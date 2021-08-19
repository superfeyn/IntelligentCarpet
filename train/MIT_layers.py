import torch.nn as nn
import torch as tr
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):

    def __init__(self, num_head, model_dim, k_dim, v_dim, dr=0.1):
        super().__init__()

        self.num_head = num_head
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.w_qs = nn.Linear(model_dim, num_head * k_dim, bias=False)
        self.w_ks = nn.Linear(model_dim, num_head * k_dim, bias=False)
        self.w_vs = nn.Linear(model_dim, num_head * k_dim, bias=False)
        self.fc = nn.Linear(num_head * v_dim, model_dim, bias=False)

        self.attention = ScaledDotProductAttention(temperature=k_dim ** 0.5)

        self.dropout = nn.Dropout(dr)
        self.layer_norm = nn.LayerNorm(model_dim, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        k_dim, v_dim, num_head = self.k_dim, self.v_dim, self.num_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, num_head, k_dim)
        k = self.w_ks(k).view(sz_b, len_k, num_head, k_dim)
        v = self.w_vs(v).view(sz_b, len_v, num_head, v_dim)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = tr.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = tr.matmul(attn, v)

        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, input_dim, hidden_dim, dr=0.1):
        super().__init__()
        self.w_1 = nn.Linear(input_dim, hidden_dim) # position-wise
        self.w_2 = nn.Linear(hidden_dim, input_dim) # position-wise
        self.layer_norm = nn.LayerNorm(input_dim, eps=1e-6)
        self.dropout = nn.Dropout(dr)

    def forward(self, x):

        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)

        return x

class EncoderLayer(nn.Module):

    def __init__(self, model_dim, inner_dim, num_head, k_dim, v_dim, dr=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(num_head, model_dim, k_dim, v_dim, dr=dr)
        self.pos_ffn = PositionwiseFeedForward(model_dim, inner_dim, dr=dr)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, model_dim, inner_dim, num_head, k_dim, v_dim, dr=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(num_head, model_dim, k_dim, v_dim, dr=dr)
        self.enc_attn = MultiHeadAttention(num_head, model_dim, k_dim, v_dim, dr=dr)
        self.pos_ffn = PositionwiseFeedForward(model_dim, inner_dim, dr=dr)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn