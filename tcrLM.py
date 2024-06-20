
import math
import blosum as bl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from encoder import *
from torch.cuda.amp import autocast

threshold = 0.5

use_cuda = torch.cuda.is_available()

n_heads= 1
n_layers = 1
seq_max_len = 34
model_data = torch.load("/data/ycp/fx/计设/ProtFlash-main/flash_protein.pt")
hyper_parameter = model_data["hyper_parameters"]
d_k = d_v = 64
d_ff = 512
#构建索引表
residue_tokens = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K',
                  'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', '<UNK>', 'PADDING_MASK', 'TOKEN_MASK']

matrix = bl.BLOSUM(50)

def transfer(y_prob, threshold=0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])

token_to_index = {}
for i, j in enumerate(residue_tokens):
    token_to_index[j] = i

f_mean = lambda l: sum(l) / len(l)


tcr_max_len = 34
d_model = 512

def calculate_perplexity(loss):
    return torch.exp(loss)

batch_size = 2048

residue_tokens = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K',
                  'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C','-']
#
def transfer(y_prob, threshold=0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])

token_to_index = {}
for i, j in enumerate(residue_tokens):
    token_to_index[j] = i

def encode_sequence_with_index(sequence_indices, matrix):
    encoded_sequence = []
    for index in sequence_indices:
        amino_acid = residue_tokens[index]
        encoded_amino_acid = [matrix[amino_acid][aa] for aa in residue_tokens]
        encoded_sequence.append(encoded_amino_acid)
    return encoded_sequence


class CustomEmbedding(nn.Module):
    def __init__(self, matrix):
        super().__init__()
        self.matrix = matrix
    def forward(self, sequence_indices):
        sequence = []
        for index in sequence_indices:
            encoded_sequence = []
            for index_tensor in index:
                index = index_tensor.item()
                amino_acid = residue_tokens[index]
                encoded_amino_acid = [self.matrix[amino_acid][aa] for aa in residue_tokens]
                encoded_sequence.append(encoded_amino_acid)
            sequence.append(encoded_sequence)
        return torch.tensor(sequence)

class Pretrain(nn.Module):
    def __init__(self, seq_max_len = 20):
        super().__init__()
        self.protflash = FLASHTransformer(hyper_parameter['dim'], hyper_parameter['num_tokens'], hyper_parameter['num_layers'], group_size=hyper_parameter['num_tokens'],
                             query_key_dim=hyper_parameter['qk_dim'], max_rel_dist=hyper_parameter['max_rel_dist'], expansion_factor=hyper_parameter['expansion_factor'])

        self.projection_mask = nn.Sequential(
            nn.Linear(512 * seq_max_len, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64,21 * seq_max_len)
        )
    def forward(self, x, lengths):
        outputs = self.protflash(x, lengths)
        seq_outputs = outputs.view(outputs.shape[0],-1)

        mask_prediction_logits = self.projection_mask(seq_outputs)
        mask_logits = mask_prediction_logits.view(batch_size, seq_max_len, 21)
        return  mask_logits

class Finetune(nn.Module):
    def __init__(self, seq_max_len=34, feature_size=512):
        super().__init__()
        self.encoder_T = FLASHTransformer(hyper_parameter['dim'], hyper_parameter['num_tokens'], hyper_parameter['num_layers'], group_size=hyper_parameter['num_tokens'],
                             query_key_dim=hyper_parameter['qk_dim'], max_rel_dist=hyper_parameter['max_rel_dist'], expansion_factor=hyper_parameter['expansion_factor'])
        self.encoder_P= FLASHTransformer(hyper_parameter['dim'], hyper_parameter['num_tokens'], hyper_parameter['num_layers'], group_size=hyper_parameter['num_tokens'],
                             query_key_dim=hyper_parameter['qk_dim'], max_rel_dist=hyper_parameter['max_rel_dist'], expansion_factor=hyper_parameter['expansion_factor'])
        self.seq_max_len = seq_max_len
        self.feature_size = feature_size
        self.classifier = nn.Linear(2 * seq_max_len * feature_size, 2)

    def forward(self, pep_inputs, tcr_inputs, pep_lengths, tcr_lengths):
        pep_outputs = self.encoder_P(pep_inputs, pep_lengths)
        tcr_outputs = self.encoder_T(tcr_inputs, tcr_lengths)
        combined_outputs = torch.cat((pep_outputs, tcr_outputs), dim=1)
        combined_outputs = combined_outputs.view(combined_outputs.shape[0], -1)
        logits = self.classifier(combined_outputs)
        return logits

