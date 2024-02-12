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

from pathlib import Path
from custom_datasets import TranslationDataset

from torch.utils.data import random_split, DataLoader

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

# hyper parameters
_seq_len = 350
_batch_size = 8

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

def get_dataset():
    raw_data = load_dataset("opus100","da-en",split="train")
    tokenizer_src = get_or_build_tokenizer(raw_data,'da')
    tokenizer_tgt = get_or_build_tokenizer(raw_data,'en')

    train_ds_size = int(0.9 * len(raw_data))
    val_ds_size = len(raw_data) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(raw_data,[train_ds_size,val_ds_size])

    train_ds = TranslationDataset(train_ds_raw,tokenizer_src,tokenizer_tgt,'da','en',_seq_len)
    val_ds = TranslationDataset(val_ds_raw,tokenizer_src,tokenizer_tgt,'da','en',_seq_len)

    for item in raw_data:
        src_ids = tokenizer_src.encode(item['translation']['da']).ids
        tgt_ids = tokenizer_tgt.encode(item['translation']['en']).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
