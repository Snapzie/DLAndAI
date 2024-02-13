import torch
import torch.nn as nn
from torch.utils.data import Dataset

def get_mask(size):
    return torch.tril(torch.ones((1,size,size),dtype=bool),diagonal=0)

class TranslationDataset(Dataset):
    def __init__(self,ds,tokenizer_src,tokenizer_tgt,seq_len,src_lang='da',tgt_lang='en'):
        super().__init__()
        self.dataset = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.seq_len = seq_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

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
            'encoder_mask': (src != self.PAD).unsqueeze(0).unsqueeze(0).int(), # (1,1,seq_len)
            'decoder_mask': (tgt != self.PAD).unsqueeze(0).int() & get_mask(tgt.size(0)), # (1, seq_len) & (1, seq_len, seq_len)
            'label': label, # (seq_len)
            'src_text': src_raw,
            'tgt_text': tgt_raw
        }

