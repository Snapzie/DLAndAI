import torch
import torch.nn as nn
import torch.nn.functional as f

# Hyper parameters
_N = 6
_d_model = 512
_seq_len = 128
_num_heads = 8
_dropout = 0
_vocab_size = 28
# Read input file
# _vocab_size = len(set(input))

class EncoderLoop(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        assert d_model % num_heads == 0,'d_model not divisionable by num_heads'
        # self.seq_len = seq_len
        self.MHA = nn.MultiheadAttention(d_model,num_heads,dropout=_dropout,batch_first=True)
        self.lin1 = nn.Linear(d_model,d_model)
        self.lin2 = nn.Linear(d_model,d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(_dropout)

    def forward(self,x):
        print(f'Encoder in: {x.shape}')
        # Attention(QW,KW,VW): x @ w <==> (seq,d_model) @ (d_model,d_model) --> (seq,d_model)
        x_residual1 = x
        #self.register_buffer('tril', torch.tril(torch.ones((self.seq_len,self.seq_len),dtype=bool),diagonal=0))
        x,attn_scores = self.MHA.forward(x,x,x) # ,attn_mask=self.tril)
        x_residual2 = self.norm1(x + x_residual1)
        x = self.dropout(f.relu(self.lin1(x_residual2)))
        x = self.lin2(x)
        return self.norm2(x + x_residual2)

class Encoder(nn.Module):
    def __init__(self,d_model,num_heads,seq_len):
        super().__init__()
        self.TokentEmbedding = nn.Embedding(_vocab_size,d_model)
        self.PosEmbedding = nn.Embedding(seq_len,d_model) #TODO: Replace with cos- and sinusoidal embedding
        self.loop = nn.Sequential(*[EncoderLoop(d_model,num_heads) for _ in range(_N)])

    def forward(self,x):
        B,T = x.shape # (batch,seq_len)
        token_embedding = self.TokentEmbedding(x) # (batch,seq_len,d_model)
        pos_embedding = self.PosEmbedding(torch.arange(T)) # (seq_len,d_model)
        x = token_embedding + pos_embedding # (batch,seq_len,d_model)
        return self.loop(x)
    
class DecoderLoop(nn.Module):
    def __init__(self,d_model,num_heads,seq_len):
        super().__init__()
        self.seq_len = seq_len
        assert d_model % num_heads == 0,'d_model not divisionable by num_heads'
        self.SA = nn.MultiheadAttention(d_model,num_heads,dropout=_dropout,batch_first=True)
        self.CA = nn.MultiheadAttention(d_model,num_heads,dropout=_dropout,batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.lin1 = nn.Linear(d_model,d_model)
        self.lin2 = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(_dropout)

    def forward(self,x,encoder_output):
        print(f'Decoder in {x.shape}')
        x_residual1 = x
        self.register_buffer('tril', torch.tril(torch.ones((self.seq_len,self.seq_len),dtype=bool),diagonal=0))
        x,attn_scores = self.SA(x,x,x,attn_mask=self.tril)
        x_residual2 = self.norm1(x + x_residual1)
        x,attn_scores = self.CA(x,encoder_output,encoder_output)
        x_residual3 = self.norm2(x + x_residual2)
        x = self.dropout(f.relu(self.lin1(x_residual3)))
        x = self.lin2(x)
        return self.norm3(x + x_residual3)

class Decoder(nn.Module):
    def __init__(self,d_model,num_heads,seq_len):
        super().__init__()
        self.TokentEmbedding = nn.Embedding(_vocab_size,d_model)
        self.PosEmbedding = nn.Embedding(seq_len,d_model) #TODO: Replace with cos- and sinusoidal embedding
        self.loop = nn.ModuleList([DecoderLoop(d_model,num_heads,seq_len) for _ in range(_N)])
    
    def forward(self,x,encoder_output):
        B,T = x.shape # (batch,seq_len)
        token_embedding = self.TokentEmbedding(x) # (batch,seq_len,d_model)
        pos_embedding = self.PosEmbedding(torch.arange(T)) # (seq_len,d_model)
        x = token_embedding + pos_embedding # (batch,seq_len,d_model)
        for decoderLoop in self.loop:
            x = decoderLoop(x,encoder_output)
        return x



encoder = Encoder(_d_model,_num_heads,_seq_len)
x = torch.randint(0,_vocab_size,(1,_seq_len))
encoder_output = encoder.forward(x)

decoder = Decoder(_d_model,_num_heads,_seq_len)
x = torch.randint(0,_vocab_size,(1,_seq_len))
decoder_output = decoder.forward(x,encoder_output)