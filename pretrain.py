import argparse
import os
import time
import datetime
import torch.nn.functional as F
import deepspeed
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from encoder import FLASHTransformer
from utils import load_hub_workaround,batchConverter
from torch.utils.data import Dataset, DataLoader
from random import randint, shuffle, random
from torch.utils.data import DataLoader

use_cuda = torch.cuda.is_available()

seq_max_len = 20
epoch_best = -1
perplexity_min = 1e12
save_dir = '/data/model/'
metric_best, ep_best = 0, -1
time_train = 0

classification_loss_fn = torch.nn.CrossEntropyLoss()
mask_prediction_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

residue_tokens = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K',
                  'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', '<UNK>', 'PADDING_MASK', 'TOKEN_MASK']

token_to_index = {}
for i, j in enumerate(residue_tokens):
    token_to_index[j] = i

f_mean = lambda l: sum(l) / len(l)

batch_size = 768
args_parser = argparse.ArgumentParser()
args_parser.add_argument('--local_rank', type=int,
                        help='Local rank passed from distributed launcher')
deepspeed.add_config_arguments(args_parser)
args = args_parser.parse_args()

num_epochs = 10
def calculate_perplexity(loss):
    return torch.exp(loss)

class MyModel(nn.Module):
    def __init__(self, protflash,seq_max_len = 20):
        super().__init__()
        self.protflash = protflash
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

def load_prot_flash_small():
    model_data = torch.load("/data/flash_protein.pt")
    hyper_parameter = model_data["hyper_parameters"]
    model = FLASHTransformer(hyper_parameter['dim'], hyper_parameter['num_tokens'], hyper_parameter['num_layers'], group_size=hyper_parameter['num_tokens'],
                             query_key_dim=hyper_parameter['qk_dim'], max_rel_dist=hyper_parameter['max_rel_dist'], expansion_factor=hyper_parameter['expansion_factor'])

    model.load_state_dict(model_data['state_dict'])
    return model

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

def label_to_onehot(label: str):
    label_map = {'tcr':[1,0], 'pep':[0,1]}
    return label_map.get(label, [0,0])

def create_antigen_dataset_from_csv_with_pandas(file_path: str):
    df = pd.read_csv(file_path)
    sequences = df['sequence'].tolist()
    labels = [label_to_onehot(label) for label in df['label'].tolist()]

    def get_length():
        return len(sequences)

    def get_item(idx):
        return str(idx), sequences[idx], labels[idx]
    return get_length, get_item

class AntigenDataset(Dataset):
    def __init__(self, file_path: str):
        self.get_length, self.get_item = create_antigen_dataset_from_csv_with_pandas(file_path)
        self.length = self.get_length()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        id_, sequence, label = self.get_item(idx)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return id_, sequence, label_tensor


def data_with_loader(type_='train',batch_size=768, use_distributed=True):
    if type_ != 'train' and type_ != 'val':
        csv_file_path = '/data/{}_set'.format(type_)
    elif type_ == 'train':
        csv_file_path = '/data/train_data.csv'
    dataset = AntigenDataset(csv_file_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,drop_last=True)
    return dataset, loader
def train_step(model_engine, train_loader):
    time_train_ep = 0
    loss_train_list = []
    perplexity_list = []
    model_engine.train()
    for row_batch in tqdm(train_loader,colour = 'yellow'):
        time_train_ep = 0
        sequence_ids, sequences, classification_labels = row_batch
        combined_data = [(id_, seq) for id_, seq in zip(sequence_ids, sequences)]
        optimizer.zero_grad(set_to_none=True)
        ids, train_batch_token, train_lengths = batchConverter(combined_data)
        train_inputs, mask_labels = mask_sequence(train_batch_token, token_to_index['TOKEN_MASK'], train_lengths)
        t1 = time.time()
        mask_prediction_logits = model_engine(train_inputs, train_lengths)
        mask_labels = torch.tensor(mask_labels)
        mask_prediction_logits = mask_prediction_logits.to(mask_labels.device)
        mask_prediction_logits = mask_prediction_logits.float()
        mask_prediction_loss = mask_prediction_loss_fn(mask_prediction_logits.view(-1,21), mask_labels.view(-1))
        time_train_ep += time.time() - t1
        model_engine.backward(mask_prediction_loss)
        model_engine.step()
        mask_prediction_loss = mask_prediction_loss.detach()
        perplexity = calculate_perplexity(mask_prediction_loss).item()
        loss_train_list.append(mask_prediction_loss.item())
        perplexity_list.append(perplexity)
    print('Fold-****Train (Ep avg): | Loss = {:.4f} | Perplexity = {:.4f} sec'.format(
                                                                                              f_mean(loss_train_list),
                                                                                              f_mean(perplexity_list)))
    return loss_train_list,perplexity_list,time_train_ep

protflash = load_prot_flash_small()
model = MyModel(protflash)
optimizer = optim.Adam([
    {'params': model.protflash.parameters(), 'lr': 1e-6},
    {'params': model.projection_mask.parameters()}
], lr=1e-4,betas=(0.9, 0.999),eps=1e-8,weight_decay=1e-5)

model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              optimizer=optimizer,
                                              model_parameters=model.parameters())

print('-----Train-----')
print('-----Generate data loader-----')
train_dataset,train_loader = data_with_loader(type_='train',   batch_size=batch_size)
for epoch in range(num_epochs):
    print('Epoch: {}/{}'.format(epoch + 1, num_epochs))
    print('dir_saver: ', save_dir)
    loss_train_list, perplexity_train_list, time_train_ep = train_step(model_engine, train_loader)
    perplexity_val_mean = f_mean(perplexity_train_list)
    if perplexity_val_mean < perplexity_min:
        perplexity_min, epoch_best = perplexity_val_mean, epoch
        print('****Saving model: Best epoch = {} | perplexity_min = {:.4f}'.format(epoch_best, perplexity_min))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_engine.save_checkpoint(save_dir)
    epoch_end_time = time.time()
print('-----Optimization Finished!-----')







