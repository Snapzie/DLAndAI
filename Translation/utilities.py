import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader, Subset

from pathlib import Path
import pandas as pd

import datasets
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

# Dataset class for translation tasks. Source language being danish and target language being english
class TranslationDataset(Dataset):
    def __init__(self,ds,tokenizer_src,tokenizer_tgt,seq_len,num_heads,src_lang='da',tgt_lang='en'):
        super().__init__()
        self.dataset = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.seq_len = seq_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.num_heads = num_heads

        self.SOS = torch.tensor([tokenizer_src.token_to_id('[SOS]')],dtype=torch.int64)
        self.EOS = torch.tensor([tokenizer_src.token_to_id('[EOS]')],dtype=torch.int64)
        self.PAD = torch.tensor([tokenizer_src.token_to_id('[PAD]')],dtype=torch.int64)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        src_raw = item['translation'][self.src_lang]
        tgt_raw = item['translation'][self.tgt_lang]
        src_enc = self.tokenizer_src.encode(src_raw).ids
        tgt_enc = self.tokenizer_tgt.encode(tgt_raw).ids

        #compute padding
        src_pad = self.seq_len - len(src_enc)
        tgt_pad = self.seq_len - len(tgt_enc)

        assert src_pad > 1 and tgt_pad > 0, "TranslationDataset: Seq_len too short / Sentence too long"

        src = torch.concat([
            self.SOS,
            torch.tensor(src_enc,dtype=torch.int64),
            self.EOS,
            torch.tensor([self.PAD] * (src_pad-2),dtype=torch.int64)
        ], dim=0)

        tgt = torch.concat([
            self.SOS,
            torch.tensor(tgt_enc,dtype=torch.int64),
            torch.tensor([self.PAD] * (tgt_pad-1), dtype=torch.int64)
        ], dim=0)

        label = torch.concat([
            torch.tensor(tgt_enc,dtype=torch.int64),
            self.EOS,
            torch.tensor([self.PAD] * (tgt_pad-1),dtype=torch.int64)
        ], dim=0)

        assert src.size(0) == self.seq_len, 'src encoding does not match seq_len'
        assert tgt.size(0) == self.seq_len, 'tgt encoding does not match seq_len'
        assert label.size(0) == self.seq_len, 'label encoding does not match seq_len'

        return {
            'encoder_input': src, # (seq_len)
            'decoder_input': tgt, # (seq_len)
            # encoder_mask wants to remove the paddings from the attention. Thus broadcasting 1D mask to get a 'block'
            # rather than a triangle as there is no time aspect in the encoder.
            'encoder_mask': (src != self.PAD).type(torch.bool), #.repeat(self.num_heads,self.seq_len,1).view(self.num_heads,self.seq_len,self.seq_len), # (num_heads,seq_len,seq_len)
            'decoder_mask': ((tgt != self.PAD).type(torch.bool) & get_mask(tgt.size(0))).repeat(self.num_heads,1,1).view(self.num_heads,self.seq_len,self.seq_len), # (seq_len) & (seq_len, seq_len) --> (num_heads,seq_len,seq_len)
            'label': label, # (seq_len)
            'src_text': src_raw,
            'tgt_text': tgt_raw
        }

def get_mask(size):
    return torch.tril(torch.ones((size,size),dtype=bool),diagonal=0)

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(ds, lang):
    tokenizer_path = Path(f'tokenizer_{lang}.json')
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(seq_len,batch_size,num_heads,ds_size,num_translations=5):
    # raw_data = load_dataset("opus100","da-en",split="validation")
    pd_data = pd.read_parquet('./validation-00000-of-00001.parquet')
    raw_data = datasets.Dataset.from_pandas(pd_data)
    train_ds_size = ds_size
    val_ds_size = len(raw_data) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(raw_data,[train_ds_size,val_ds_size])

    src_tokenizer = get_or_build_tokenizer(train_ds_raw,'da')
    tgt_tokenizer = get_or_build_tokenizer(train_ds_raw,'en')

    print('Computing dataset sizes...')
    max_len_src = 0
    max_len_tgt = 0
    for item in train_ds_raw:
        src_ids = src_tokenizer.encode(item['translation']['da']).ids
        tgt_ids = tgt_tokenizer.encode(item['translation']['en']).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    print(f'Src vocab size: {src_tokenizer.get_vocab_size()}')
    print(f'Tgt vocab size: {tgt_tokenizer.get_vocab_size()}')
    print(f'Size of dataset: {len(train_ds_raw)}')

    train_ds = TranslationDataset(train_ds_raw,src_tokenizer,tgt_tokenizer,seq_len,num_heads)
    val_ds = TranslationDataset(val_ds_raw,src_tokenizer,tgt_tokenizer,seq_len,num_heads)

    # train_ds = Subset(train_ds,torch.arange(ds_size))
    tranlation_set = Subset(train_ds,torch.arange(num_translations))

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    translation_dataloader = DataLoader(tranlation_set, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, translation_dataloader, src_tokenizer, tgt_tokenizer 


    # raw_data = load_dataset("opus100","da-en",split="train")
    # src_tokenizer = get_or_build_tokenizer(raw_data,'da')
    # tgt_tokenizer = get_or_build_tokenizer(raw_data,'en')

    # # Filter the data to ensure all sentences are smaller than seq_len
    # print('Filtering size of dataset...')
    # raw_data = raw_data.filter(lambda x: len(src_tokenizer.encode(x['translation']['da'])) < seq_len and len(src_tokenizer.encode(x['translation']['en'])) < seq_len)

    # # print('Computing dataset sizes...')
    # # max_len_src = 0
    # # max_len_tgt = 0
    # # for item in raw_data:
    # #     src_ids = src_tokenizer.encode(item['translation']['da']).ids
    # #     tgt_ids = tgt_tokenizer.encode(item['translation']['en']).ids
    # #     max_len_src = max(max_len_src, len(src_ids))
    # #     max_len_tgt = max(max_len_tgt, len(tgt_ids))

    # # print(f'Max length of source sentence: {max_len_src}')
    # # print(f'Max length of target sentence: {max_len_tgt}')
    # # print(f'Size of dataset: {len(raw_data)}')

    # # Create training and validation sets
    # train_ds_size = int(0.9 * len(raw_data))
    # val_ds_size = len(raw_data) - train_ds_size
    # train_ds_raw, val_ds_raw = random_split(raw_data,[train_ds_size,val_ds_size])

    # train_ds = TranslationDataset(train_ds_raw,src_tokenizer,tgt_tokenizer,seq_len,num_heads)
    # val_ds = TranslationDataset(val_ds_raw,src_tokenizer,tgt_tokenizer,seq_len,num_heads)

    # train_ds = Subset(train_ds,torch.arange(ds_size))
    # tranlation_set = Subset(train_ds,torch.arange(num_translations))

    # train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    # translation_dataloader = DataLoader(tranlation_set, batch_size=1, shuffle=False)

    # return train_dataloader, val_dataloader, translation_dataloader, src_tokenizer, tgt_tokenizer 