import torch
import torch.nn as nn
import math
import torch.nn.functional as F
# from entmax import entmax15

device='cuda'

class MultiHead_Attn_Layer(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super(MultiHead_Attn_Layer, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.fc_Q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_K = nn.Linear(hidden_dim, hidden_dim)
        self.fc_V = nn.Linear(hidden_dim, hidden_dim)
        self.fc_O = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None, topk=None):                                    # [query] = [batch_size, query_len, hidden_dim] [key] = [batch_size, key_len, hidden_dim] [value] = [batch_Size, value_len, hidden_dim]
        batch_size = query.shape[0]                                           
        # print(query.shape)
        Q = self.fc_Q(query)                                                            # [Q] = [batch_size, query_len, hidden_dim]   
        K = self.fc_K(key)                                                              # [K] = [batch_size, key_len, hidden_dim]
        V = self.fc_V(value)                                                            # [V] = [batch_size, value_len, hidden_dim]
        # print(Q.shape)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)     # [Q] = [batch_size, num_heads, query_len, head_dim]
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)     # [K] = [batch_size, num_heads, key_len, head_dim]
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)     # [V] = [batch_size, num_heads, value_len, head_dim]
        
        # print(Q.shape)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        # print(energy.shape)
        if mask is not None:    
            MASKING_VALUE = -1e+30 #if energy.dtype == torch.float32 else -1e+4                                                        # [energy] = [batch_size, num_heads, query_len, key_len]  
            energy = energy.masked_fill(mask == False, MASKING_VALUE)                               
        
        attention = torch.softmax(energy, dim = -1)                                     # [attention] = [batch_size, num_heads, query_len, key_len]
        
        if topk:
            topk = torch.topk(attention, dim = -1, k = topk)
            mask = torch.zeros(attention.size()).to(attention.device)
            mask.scatter_(3, topk.indices, 1)
            attention = attention * mask 
        
        x = torch.matmul(self.dropout(attention), V)                                    # [x] = [batch_size, num_heads, query_len, head_dim]
        x = x.permute(0, 2, 1, 3).contiguous()                                          # [x] = [batch_size, query_len, num_heads, head_dim]
        # Can avoid contiguous() if we use .reshape instead of .view in the next line
        out = self.fc_O(x.view(batch_size, -1, self.hidden_dim))                        # [out] = [batch_size, query_len, hidden_dim]   
        return out, attention

class Postn_Feed_Fwrd(nn.Module):
    def __init__(self, hidden_dim, pff_dim, dropout):
        super(Postn_Feed_Fwrd, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, pff_dim)
        self.fc2 = nn.Linear(pff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):                                                   # input = [batch_size, seq_len, hidden_dim]
        out = torch.relu(self.fc1(input))                                       # out = [batch_size, seq_len, pff_dim]
        out = self.fc2(self.dropout(out))                                       # out = [batch_size, seq_len, hidden_dim] 
        return out

class Encoder_Layer(nn.Module):
    def __init__(self, hidden_dim, n_heads, pff_dim, dropout):
        super(Encoder_Layer, self).__init__()
        self.self_attn = MultiHead_Attn_Layer(hidden_dim, n_heads, dropout)    
        self.pff = Postn_Feed_Fwrd(hidden_dim, pff_dim, dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.pff_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):                                           # src = [batch_size, src_len, hidden_dim]  src_mask = [batch_size, src_len]
        
        attn_out, _ = self.self_attn(src, src, src, src_mask)                   # attn_out = [batch_size, src_len, hidden_dim]
        inter_out = self.attn_norm(self.dropout(attn_out) + src)                # inter_out = [batch_size, src_len, hidden_dim]
        pff_out = self.pff(inter_out)                                           # pff_out = [batch_size, src_len, hidden_dim]
        out = self.pff_norm(self.dropout(pff_out) + inter_out)                  # out = [batch_size, src_len, hidden_dim]
        return out

class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, pff_dim, dropout):
        super(CrossModalAttention, self).__init__()
        self.self_attn = MultiHead_Attn_Layer(hidden_dim, n_heads, dropout)    
        self.pff = Postn_Feed_Fwrd(hidden_dim, pff_dim, dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.pff_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, src_mask, topk=None):                             # src = [batch_size, src_len, hidden_dim]  src_mask = [batch_size, src_len]
        
        attn_out, _ = self.self_attn(query, key, value, src_mask, topk)               # attn_out = [batch_size, src_len, hidden_dim]
        inter_out = self.attn_norm(self.dropout(attn_out) + query)              # inter_out = [batch_size, src_len, hidden_dim]
        pff_out = self.pff(inter_out)                                           # pff_out = [batch_size, src_len, hidden_dim]
        out = self.pff_norm(self.dropout(pff_out) + inter_out)                  # out = [batch_size, src_len, hidden_dim]
        return out
    
class Decoder_Layer(nn.Module):
    def __init__(self, hidden_dim, n_heads, pff_dim, dropout):
        super(Decoder_Layer, self).__init__()
        self.self_attn = MultiHead_Attn_Layer(hidden_dim, n_heads, dropout)
        self.cross_attn = MultiHead_Attn_Layer(hidden_dim, n_heads, dropout)
        self.pff = Postn_Feed_Fwrd(hidden_dim, pff_dim, dropout)
        self.attn_norm1 = nn.LayerNorm(hidden_dim)
        self.attn_norm2 = nn.LayerNorm(hidden_dim)
        self.pff_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, trg_mask, enc_out, src_mask):                            # trg = [batch_size, trg_len, hidden_dim] trg_mask = [batch_size, trg_len] enc_out = [batch_size, src_len, hidden_dim] src_mask = [batch_size, src_len]
        sattn_out, _ = self.self_attn(trg, trg, trg, trg_mask)                      # satten_out = [batch_size, trg_len, hidden_dim]
        inter_out1 = self.attn_norm1(self.dropout(sattn_out) + trg)                 # inter_out1 = [batch_size, trg_len, hidden_dim]  
        cattn_out, attn = self.cross_attn(inter_out1, enc_out, enc_out, src_mask)   # cattn_out = [batch_size, trg_len, hidden_dim] attn = [batch_size, num_heads, query_len, key_len]
        inter_out2 = self.attn_norm2(self.dropout(cattn_out) + inter_out1)          # inter_out2 = [batch_size, trg_len, hidden_dim]
        pff_out = self.pff(inter_out2)                                              # pff_out = [batch_size, trg_len, hidden_dim]
        out = self.pff_norm(self.dropout(pff_out) + inter_out2)                     # out = [batch_size, trg_len, hidden_dim]
        return out, attn


class Decoder(nn.Module):
    def __init__(self, tok_vocab_size, pos_vocab_size, hidden_dim, dec_heads, dec_pff_dim, num_layers, dec_dropout):
        super(Decoder, self).__init__()
        self.tok_embedding = nn.Embedding(tok_vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(pos_vocab_size, hidden_dim) #pos_vocab_size = Max possible sequence length
        self.dec_layers = nn.ModuleList([Decoder_Layer(hidden_dim, dec_heads, dec_pff_dim, dec_dropout) for i in range(num_layers)])
        self.fc = nn.Linear(hidden_dim, tok_vocab_size)
        self.scale = math.sqrt(hidden_dim)
        self.dropout = nn.Dropout(dec_dropout)

    def forward(self, trg, trg_mask, enc_out, src_mask):                                    # trg = [batch_size, trg_len] trg_mask = [batch_size, trg_len] enc_out = [] src_mask = []
        batch_size = trg.shape[0] 
        trg_len = trg.shape[1]
        tok_embed = self.tok_embedding(trg)                                                 # tok_embed = [batch_size, trg_len, hidden_dim]
        pos_tensor = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(device) # pos_tensor = [batch_size, trg_len]
        pos_embed = self.pos_embedding(pos_tensor)                                          # pos_embed = [batch_size, trg_len, hidden_dim]
        dec_embed = self.dropout(tok_embed * self.scale + pos_embed)                        # dec_embed = [batch_size, trg_len, hidden_dim]
        dec_state = dec_embed
        for dec_layer in self.dec_layers:
            dec_state, attention = dec_layer(dec_state, trg_mask, enc_out, src_mask)        # dec_state = [batch_size, trg_len, hidden_dim] attention = [batch_size, num_heads, query_len, key_len]
        out = self.fc(dec_state)                                                            # out = [batch_size, tok_vocab_size]
        return out, attention
