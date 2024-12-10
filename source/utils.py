import torch.utils.data as Data
from random import randint
from typing import Sequence, Tuple
import pathlib
import urllib
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np


residue_tokens = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K',
                  'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', '<UNK>', 'PADDING_MASK', 'TOKEN_MASK']
token_to_index = {}
for i, j in enumerate(residue_tokens):
    token_to_index[j] = i

seq_max_len = 20

f_mean = lambda l: sum(l) / len(l)

def transfer(y_prob, threshold=0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])

def calculate_perplexity(loss):
    return torch.exp(loss)

def pad_to_fixed_length(batch_token, fixed_length, padding_value):
    padded_tokens = torch.full((len(batch_token), fixed_length), fill_value=padding_value, dtype=torch.long)
    for i, tokens in enumerate(batch_token):
        length = min(len(tokens), fixed_length)
        padded_tokens[i, :length] = torch.tensor(tokens[:length], dtype=torch.long)
    return padded_tokens

def mask_sequence(token_indices, mask_token_idx, lengths):
    label_list = []
    token_list = token_indices.tolist()
    lengths_list = lengths.tolist()
    for idx, (seq, length) in enumerate(zip(token_list, lengths_list)):
        labels = [-100] * len(seq)
        mask_length = randint(3, 5)
        if length > 20:
            length = 20
        mask_start = randint(0, max(1, length - mask_length))
        mask_end = mask_start + mask_length
        for i in range(mask_start, min(mask_end, length)):
            labels[i] = seq[i]
            seq[i] = mask_token_idx
        label_list.append(labels)
        token_list[idx] = seq
    token_indices = torch.tensor(token_list)
    label_tensors = torch.tensor(label_list)
    return token_indices, label_tensors

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


class MyDataSet(Data.Dataset):
    def __init__(self, pep_inputs, tcr_inputs, labels,pep_length,tcr_length):
        super(MyDataSet, self).__init__()
        self.pep_inputs = pep_inputs
        self.tcr_inputs = tcr_inputs
        self.labels = labels
        self.pep_length = pep_length
        self.tcr_length = tcr_length

    def __len__(self):
        return self.pep_inputs.shape[0]

    def __getitem__(self, idx):
        return self.pep_inputs[idx], self.tcr_inputs[idx], self.labels[idx],self.pep_length[idx],self.tcr_length[idx]

def create_antigen_dataset_from_csv_with_pandas(file_path: str):
    df = pd.read_csv(file_path)
    sequences = df['sequence'].tolist()

    def get_length():
        return len(sequences)

    def get_item(idx):
        return str(idx), sequences[idx]
    return get_length, get_item

class AntigenDataset(Data.Dataset):
    def __init__(self, file_path: str):
        self.get_length, self.get_item = create_antigen_dataset_from_csv_with_pandas(file_path)
        self.length = self.get_length()

    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        id_, sequence = self.get_item(idx)
        return id_, sequence

def performances_to_pd(performances_list):
    metrics_name = ['roc_auc', 'accuracy', 'mcc', 'f1', 'aupr', 'sensitivity', 'specificity', 'precision', 'recall']

    performances_pd = pd.DataFrame(performances_list, columns=metrics_name)
    performances_pd.loc['mean'] = performances_pd.mean(axis=0)
    performances_pd.loc['std'] = performances_pd.std(axis=0)

    return performances_pd
