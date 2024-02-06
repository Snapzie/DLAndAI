import torch
import torch.nn as nn
import torch.nn.functional as f

# Hyper parameters
_N = 1
_d_model = 512
_seq_len = 128
_num_heads = 8
_dropout = 0
_vocab_size = 28
# Read input file
# _vocab_size = len(set(input))

class EncoderLoop(nn.Module):
    def __init__(self,d_model,num_heads,seq_len):
        super().__init__()
        assert d_model % num_heads == 0,'d_model not divisionable by num_heads'
        self.seq_len = seq_len
        self.MHA = nn.MultiheadAttention(d_model,num_heads,dropout=_dropout,batch_first=True)
        self.lin1 = nn.Linear(d_model,d_model)
        self.lin2 = nn.Linear(d_model,d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(_dropout)

    def forward(self,x):
        # Attention(QW,KW,VW): x @ w <==> (seq,d_model) @ (d_model,d_model) --> (seq,d_model)
        x_residual1 = x
        self.register_buffer('tril', torch.tril(torch.ones((self.seq_len,self.seq_len),dtype=bool),diagonal=0))
        x,attn_scores = self.MHA.forward(x,x,x,attn_mask=self.tril)
        x_residual2 = self.norm1(x + x_residual1)
        x = self.dropout(f.relu(self.lin1(x_residual2)))
        x = self.lin2(x)
        x = self.norm2(x + x_residual2)

class Encoder(nn.Module):
    def __init__(self,d_model,num_heads,seq_len):
        super().__init__()
        self.TokentEmbedding = nn.Embedding(d_model,_vocab_size)
        self.PosEmbedding = nn.Embedding(_seq_len,d_model) #TODO: Replace with cos- and sinusoidal embedding
        self.loop = nn.Sequential(*[EncoderLoop(d_model,num_heads,seq_len) for _ in range(_N)])

    def forward(self,x):
        #token_embedding = self.TokentEmbedding(x)
        #os_embedding = self.PosEmbedding(x)
        # x = token_embedding + pos_embedding
        x = self.loop(x)

model = Encoder(_d_model,_num_heads,_seq_len)
x = torch.randn((1,_seq_len,_d_model))
model.forward(x)