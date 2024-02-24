import torch
import torch.nn as nn
import torch.nn.functional as f
import math
from utilities import get_mask
# Hyper parameters
_N = 6
_d_model = 512
_seq_len = 128
_num_heads = 8
_dropout = 0
_vocab_size = 28
# Read input file
# _vocab_size = len(set(input))

class PositionalEmbedding(nn.Module):
    def __init__(self,d_model,seq_len,dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe=torch.zeros(seq_len,d_model)
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1) # (seq_len,1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1,seq_len,d_model)
        self.register_buffer('pe',pe)

    def forward(self,x):
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False) # (batch,seq_len,d_model)
        return self.dropout(x)

class EncoderLoop(nn.Module):
    def __init__(self,d_model,num_heads,dropout):
        super().__init__()
        assert d_model % num_heads == 0,'d_model not divisionable by num_heads'
        # self.seq_len = seq_len
        self.MHA = nn.MultiheadAttention(d_model,num_heads,dropout=dropout,batch_first=True)
        self.lin1 = nn.Linear(d_model,d_model)
        self.lin2 = nn.Linear(d_model,d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask):
        # Attention(QW,KW,VW): x @ w <==> (seq,d_model) @ (d_model,d_model) --> (seq,d_model)
        x_residual1 = x
        #self.register_buffer('tril', torch.tril(torch.ones((self.seq_len,self.seq_len),dtype=bool),diagonal=0))
        x,attn_scores = self.MHA.forward(x,x,x,key_padding_mask=mask) # ,attn_mask=self.tril)
        x_residual2 = self.norm1(x + x_residual1)
        x = self.dropout(f.relu(self.lin1(x_residual2)))
        x = self.lin2(x)
        return self.norm2(x + x_residual2)

class Encoder(nn.Module):
    def __init__(self,d_model,num_heads,seq_len,vocab_size,dropout=0.1,N=4):
        super().__init__()
        self.TokentEmbedding = nn.Embedding(vocab_size,d_model)
        self.PosEmbedding = PositionalEmbedding(d_model,seq_len,dropout)
        self.loop = nn.ModuleList([EncoderLoop(d_model,num_heads,dropout=dropout) for _ in range(N)])

    def forward(self,x,mask):
        B,T = x.shape # (batch,seq_len)
        token_embedding = self.TokentEmbedding(x) # (batch,seq_len,d_model)
        x = self.PosEmbedding(token_embedding) # (batch,seq_len,d_model)
        for encoderLoop in self.loop:
            x = encoderLoop(x,mask)
        return x
    
class DecoderLoop(nn.Module):
    def __init__(self,d_model,num_heads,seq_len,dropout):
        super().__init__()
        self.seq_len = seq_len
        assert d_model % num_heads == 0,'d_model not divisionable by num_heads'
        self.SA = nn.MultiheadAttention(d_model,num_heads,dropout=dropout,batch_first=True)
        self.CA = nn.MultiheadAttention(d_model,num_heads,dropout=dropout,batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.lin1 = nn.Linear(d_model,d_model)
        self.lin2 = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,encoder_output,tgt_mask,src_mask):
        # Attention(QW,KW,VW): x @ w <==> (seq,d_model) @ (d_model,d_model) --> (seq,d_model)
        x_residual1 = x
        x,attn_scores = self.SA(x,x,x,attn_mask=tgt_mask)
        x_residual2 = self.norm1(x + x_residual1)
        x,attn_scores = self.CA(x_residual2,encoder_output,encoder_output,key_padding_mask=src_mask)
        x_residual3 = self.norm2(x + x_residual2)
        x = self.dropout(f.relu(self.lin1(x_residual3)))
        x = self.lin2(x)
        return self.norm3(x + x_residual3)

class Decoder(nn.Module):
    def __init__(self,d_model,num_heads,seq_len,vocab_size,dropout=0.1,N=4):
        super().__init__()
        self.TokentEmbedding = nn.Embedding(vocab_size,d_model)
        self.PosEmbedding = PositionalEmbedding(d_model,seq_len,dropout)
        self.loop = nn.ModuleList([DecoderLoop(d_model,num_heads,seq_len,dropout=dropout) for _ in range(N)])
    
    def forward(self,x,encoder_output,tgt_mask,src_mask):
        B,T = x.shape # (batch,seq_len)
        token_embedding = self.TokentEmbedding(x) # (batch,seq_len,d_model)
        x = self.PosEmbedding(token_embedding) # (batch,seq_len,d_model)
        for decoderLoop in self.loop:
            x = decoderLoop(x,encoder_output,tgt_mask,src_mask)
        return x

class Transformer(nn.Module):
    def __init__(self,d_model,num_heads,seq_len,vocab_size_src,vocab_size_tgt,dropout=0.1,N=4):
        super().__init__()
        self.encoder = Encoder(d_model,num_heads,seq_len,vocab_size_src,dropout,N)
        self.decoder = Decoder(d_model,num_heads,seq_len,vocab_size_tgt,dropout,N)
        self.projection = nn.Linear(d_model,vocab_size_tgt)
    
    def Encode(self,x,mask):
        return self.encoder.forward(x,mask)

    def Decode(self,x,encoder_output,tgt_mask,src_mask):
        return self.decoder.forward(x,encoder_output,tgt_mask,src_mask)
    
    def Projection(self,x):
        return self.projection(x)