import torch
import torch.nn as nn
import torch.functional as f

class InputEmbeddings(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

class Encoder(nn.Module):
    def __init__(self,N,d_model,seq_len,num_heads,dropout=0):
        super().__init__()
        assert(d_model % num_heads == 0,'d_model not divisionable by num_heads')
        self.N = N
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.dropout = dropout

        self.MHA = nn.MultiheadAttention(self.d_model,self.num_heads,dropout=self.dropout,batch_first=True)
        self.FF = nn.Linear(self.d_model,self.d_model)
        self.norm1 = nn.BatchNorm1d(self.d_model)

    def forward(self,x):
        # Attention(QW,KW,VW)
        # x @ w: (seq,d_model) @ (d_model,d_model) --> (seq,d_model)
        self.register_buffer('tril', torch.tril(torch.ones((self.seq_len,self.d_model),dtype=bool),diagonal=0))
        x,attn_scores = self.MHA.forward(x,x,x,attn_mask=self.tril)
