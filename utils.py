from typing import Sequence, Tuple, List, Union
from torch.nn.utils.rnn import pad_sequence
import torch
import pathlib
import urllib
import torch
from random import randint, shuffle, random
from torch.utils.data import DataLoader
from torch.optim import Adam


residue_tokens = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K',
                  'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', '<UNK>', 'PADDING_MASK', 'TOKEN_MASK']

token_to_index = {}
for i, j in enumerate(residue_tokens):
    token_to_index[j] = i

seq_max_len = 20


def pad_to_fixed_length(batch_token, fixed_length, padding_value):
    padded_tokens = torch.full((len(batch_token), fixed_length), fill_value=padding_value, dtype=torch.long)
    for i, tokens in enumerate(batch_token):
        length = min(len(tokens), fixed_length)
        padded_tokens[i, :length] = torch.tensor(tokens[:length], dtype=torch.long)
    return padded_tokens


def batchConverter(raw_batch: Sequence[Tuple[str, str]]):
    ids = [item[0] for item in raw_batch]
    seqs = [item[1] for item in raw_batch]
    lengths = torch.tensor([len(item[1]) for item in raw_batch])
    batch_token = []
    for seq in seqs:
        batch_token.append(torch.tensor([token_to_index.get(i, token_to_index["<UNK>"]) for i in seq]))
    padding_value = token_to_index['PADDING_MASK']
    batch_token = pad_to_fixed_length(batch_token, seq_max_len, padding_value)
    return ids, batch_token, lengths


def load_hub_workaround(url):
    try:
        data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")
    except RuntimeError:
        fn = pathlib.Path(url).name
        data = torch.load(
            f"{torch.hub.get_dir()}/checkpoints/{fn}",
            map_location="cpu",
        )
    except urllib.error.HTTPError as e:
        raise Exception(f"Could not load {url}, check your network!")
    return data