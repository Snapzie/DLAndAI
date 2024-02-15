def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": 'opus_books',
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

# Inpired by https://www.youtube.com/watch?v=ISNdQcPhsts&ab_channel=UmarJamil

import torch
import torch.nn as nn
from tqdm import tqdm
from utilities import get_dataset
from Models import Transformer
# from .utilities import get_dataset
# from .Models import Transformer

# hyper parameters
_seq_len = 350
_batch_size = 16
_d_model = 512
_num_heads = 8
_ds_size_cap = 250000
lr = 1e-4
label_smoothing = 0.1
num_epochs = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

print('Loading data...')
X_train, X_val, src_tokenizer, tgt_tokenizer = get_dataset(_seq_len,_batch_size,_num_heads,_ds_size_cap)
print('Loading model...')
model = Transformer(_d_model,_num_heads,_seq_len,src_tokenizer.get_vocab_size(),tgt_tokenizer.get_vocab_size()).to(device)
print('Doing setup...')
loss_fn = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id('[PAD]'),label_smoothing=label_smoothing).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=lr,eps=1e-9)

assert _batch_size % _ds_size_cap == 0, "Last batch will be missing data and result in invalid shape size (batch_size % ds_size_cap =/= 0)"

for epoch in range(num_epochs):
    torch.cuda.empty_cache()
    model.train()
    batch_iterator = tqdm(X_train,desc=f'Processing Epoch {epoch:02d}')
    for batch in batch_iterator:
        encoder_input = batch['encoder_input'].to(device) # (batch,seq_len)
        decoder_input = batch['decoder_input'].to(device) # (batch,seq_len)
        encoder_mask = batch['encoder_mask'].view(_batch_size*_num_heads,_seq_len,_seq_len).to(device) # (batch*num_heads,seq_len,seq_len)
        decoder_mask = batch['decoder_mask'].view(_batch_size*_num_heads,_seq_len,_seq_len).to(device) # (batch*num_heads,seq_len,seq_len)

        encoder_output = model.Encode(encoder_input,encoder_mask)
        decoder_output = model.Decode(decoder_input,encoder_output,decoder_mask)
        predictions = model.Projection(decoder_output)

        y = batch['label'].to(device)
        loss = loss_fn(predictions.view(-1,tgt_tokenizer.get_vocab_size()), y.view(-1))
        batch_iterator.set_postfix({'loss': f'{loss.item():6.3f}'})

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)