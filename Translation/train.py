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
import numpy as np
import torch.nn as nn
from torch.utils.data import Subset
import torch.nn.functional as F
from tqdm import tqdm
from utilities import get_dataset,get_mask
from Models import Transformer
# from .utilities import get_dataset
# from .Models import Transformer

# hyper parameters
_seq_len = 350
_batch_size = 16
_d_model = 512
_num_heads = 8
_ds_size_cap = 200
_lr = 1e-4
_label_smoothing = 0.05
_num_epochs = 100000
_num_translations = 5
_print_per = 100
_N = 3
_dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

print('Loading data...')
X_train, X_val, translation_set, src_tokenizer, tgt_tokenizer = get_dataset(_seq_len,_batch_size,_num_heads,_ds_size_cap,num_translations=_num_translations)
print('Loading model...')
model = Transformer(_d_model,_num_heads,_seq_len,src_tokenizer.get_vocab_size(),tgt_tokenizer.get_vocab_size(),N=_N,dropout=_dropout).to(device)
print('Doing setup...')
weights = torch.ones((tgt_tokenizer.get_vocab_size()))
weights[tgt_tokenizer.token_to_id('[EOS]')] = 0.005
loss_fn = nn.CrossEntropyLoss(weight=weights,ignore_index=src_tokenizer.token_to_id('[PAD]'),label_smoothing=_label_smoothing).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=_lr,eps=1e-9)

for epoch in range(_num_epochs):
    # torch.cuda.empty_cache()
    model.train()
    batch_iterator = tqdm(X_train,desc=f'Processing Epoch {epoch:02d}')
    for batch in batch_iterator:
        encoder_input = batch['encoder_input'].to(device) # (batch,seq_len)
        # print(encoder_input)
        decoder_input = batch['decoder_input'].to(device) # (batch,seq_len)
        # print(decoder_input)
        encoder_mask = batch['encoder_mask'].to(device) # (1,seq_len)
        decoder_mask = batch['decoder_mask'].view(-1,_seq_len,_seq_len).to(device) # (batch*num_heads,seq_len,seq_len)
        # decoder_mask = torch.where(decoder_mask == True,0,-np.inf)

        encoder_output = model.Encode(encoder_input,encoder_mask)
        decoder_output = model.Decode(decoder_input,encoder_output,decoder_mask,encoder_mask)
        # print(decoder_output)
        predictions = model.Projection(decoder_output)

        y = batch['label'].to(device)
        loss = loss_fn(predictions.view(-1,tgt_tokenizer.get_vocab_size()), y.view(-1))
        batch_iterator.set_postfix({'loss': f'{loss.item():6.6f}'})

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # optimizer.zero_grad(set_to_none=True)
        
        

    # Translate _num_translations number of validation samples for visual inspection
    if epoch % _print_per == 0:
        model.eval()
        with torch.no_grad():
            print("Eval")
            for X in translation_set:
                encoder_input = X['encoder_input'].to(device) # (batch,seq_len)
                # print(encoder_input)
                encoder_mask = X['encoder_mask'].to(device) # (1,seq_len)
                # print(encoder_mask.shape)
                src_sentence = X['src_text']
                tgt_sentence = X['tgt_text']
                res_sentence = torch.empty(1,_seq_len).fill_(tgt_tokenizer.token_to_id('[PAD]')).type_as(encoder_input).to(device) # (1,seq_len)
                # print(res_sentence.shape)
                res_sentence[0,0] = tgt_tokenizer.token_to_id('[SOS]') # (1,seq_len)
                # print(res_sentence)

                encoder_output = model.Encode(encoder_input,encoder_mask)
                # print(encoder_output)
                for i in range(_seq_len-1): # no larger sequences than seq_len
                    decoder_mask = (res_sentence != tgt_tokenizer.token_to_id('[PAD]')).type(torch.bool) & get_mask(res_sentence.size(1)).type(torch.bool).to(device) # (seq_len,seq_len)
                    decoder_output = model.Decode(res_sentence,encoder_output,decoder_mask,encoder_mask) # (1,seq_len,d_model)
                    # print(decoder_output)
                    # proj = model.Projection(decoder_output[:,i+1]) # (1,d_model) --> (1,vocab_size)
                    proj = model.Projection(decoder_output) # (1,seq_len,d_model) --> (1,seq_len,vocab_size)
                    probs = F.softmax(proj,dim=2)[:,i] # (1,1,vocab_size)
                    # _, next_word = torch.max(proj,dim=1)
                    _,next_word = torch.max(probs,dim=1)
                    res_sentence[0,i+1] = next_word.item()
                    # next = torch.empty(1,_seq_len).type_as(encoder_input).fill_(tgt_tokenizer.token_to_id('[PAD]'))
                    # next[0,i] = next_word.item().to(device)
                    # res_sentence = torch.cat( # (1,seq)
                    #     [res_sentence, next], dim=1
                    # )
                    print(torch.topk(probs,5).values)
                    print(torch.topk(probs,5).indices)
                    # print(tgt_tokenizer.id_to_token(next_word.item()))
                    if next_word == tgt_tokenizer.token_to_id('[EOS]'):
                        break
                # print(f'Raw res: {res_sentence}')
                res_sentence = tgt_tokenizer.decode(res_sentence.squeeze(0).detach().cpu().numpy())
                print(f'Src: {src_sentence}')
                print(f'tgt: {tgt_sentence}')
                print(f'Translation: {res_sentence}')
                print(''.join(['-']*10))


        
        
