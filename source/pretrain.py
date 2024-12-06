import argparse
import sys
import os
# 将项目根目录添加到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time

import deepspeed

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


from utils import *
from models.tcrLM import *

use_cuda = torch.cuda.is_available()

batch_size = 618
epoch_best = -1
perplexity_min = 1e12
save_dir = '../pretrained_tcrLM'
checkpoint_save_dir = '../checkpoint'
metric_best, ep_best = 0, -1
time_train = 0

mask_prediction_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

residue_tokens = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K',
                  'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', '<UNK>', 'PADDING_MASK', 'TOKEN_MASK']

token_to_index = {}
for i, j in enumerate(residue_tokens):
    token_to_index[j] = i

f_mean = lambda l: sum(l) / len(l)

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--local_rank', type=int,
                        help='Local rank passed from distributed launcher')
args_parser.add_argument('--master_port', type=int, default=29500, help='Master port for distributed training')
deepspeed.add_config_arguments(args_parser)
args = args_parser.parse_args()
num_epochs = 2

def train_step(model_engine, train_loader):
    """
    This function executes a single training step for a model, including forward pass,
    loss calculation, backward pass, optimizer step, and tracking metrics like loss
    and perplexity. It also saves model checkpoints periodically during training.

    Args:
        model_engine (torch.nn.Module): The model used for training.
        train_loader (DataLoader): DataLoader that provides batches of training data.

    Returns:
        tuple: A tuple containing three elements:
            - loss_train_list (list): A list of loss values recorded during training.
            - perplexity_list (list): A list of perplexity values recorded during training.
            - time_train_ep (float): Total training time for the epoch.
    """
    time_train_ep = 0
    loss_train_list = []
    perplexity_list = []
    model_engine.train()
    client_sd = {}
    step = 0
    for row_batch in tqdm(train_loader,colour = 'yellow'):
        sequence_ids, sequences, classification_labels = row_batch
        combined_data = [(id_, seq) for id_, seq in zip(sequence_ids, sequences)]
        optimizer.zero_grad(set_to_none=True)
        ids, train_batch_token, train_lengths = batchConverter(combined_data)
        train_inputs, mask_labels = mask_sequence(train_batch_token, token_to_index['TOKEN_MASK'], train_lengths)
        t1 = time.time()
        train_inputs = train_inputs.to(model_engine.device)
        mask_prediction_logits = model_engine(train_inputs, train_lengths)
        mask_labels = torch.tensor(mask_labels)
        mask_prediction_logits = mask_prediction_logits.to(mask_labels.device)
        mask_prediction_logits = mask_prediction_logits.float()
        mask_prediction_loss = mask_prediction_loss_fn(mask_prediction_logits.view(-1,20), mask_labels.view(-1))
        time_train_ep += time.time() - t1
        model_engine.backward(mask_prediction_loss)
        model_engine.step()
        mask_prediction_loss = mask_prediction_loss.detach()
        perplexity = calculate_perplexity(mask_prediction_loss).item()
        loss_train_list.append(mask_prediction_loss.item())
        perplexity_list.append(perplexity)
        print('Fold-****Train (Ep avg): | Loss = {:.4f} | Perplexity = {:.4f} sec'.format(mask_prediction_loss,perplexity))
        # Save checkpoint
        if step % 200 == 0:
            client_sd['step'] = step
            ckpt_id = mask_prediction_loss.item()
            if not os.path.exists(checkpoint_save_dir):
                os.makedirs(checkpoint_save_dir)
            model_engine.save_checkpoint(checkpoint_save_dir, ckpt_id)

            client_sd_path = os.path.join(checkpoint_save_dir, f"client_state_{ckpt_id}.pt")
            torch.save(client_sd, client_sd_path)
        step += 1
    return loss_train_list,perplexity_list,time_train_ep

def val_step(model_engine, val_loader):
    """
    This function performs a validation step for the model. It calculates the loss and
    perplexity for each batch in the validation dataset, and tracks the total time taken
    for the validation epoch. The model is set to evaluation mode, which disables dropout
    and batch normalization.

    Args:
        model_engine (torch.nn.Module): The trained model used for evaluation.
        val_loader (DataLoader): DataLoader that provides batches of validation data.

    Returns:
        tuple: A tuple containing three elements:
            - loss_val_list (list): A list of loss values recorded during validation.
            - perplexity_list (list): A list of perplexity values recorded during validation.
            - time_val_ep (float): Total validation time for the epoch.
    """
    time_val_ep = 0
    loss_val_list = []
    perplexity_list = []
    model_engine.eval()
    for row_batch in tqdm(val_loader,colour = 'yellow'):
        sequence_ids, sequences, classification_labels = row_batch
        combined_data = [(id_, seq) for id_, seq in zip(sequence_ids, sequences)]
        ids, val_batch_token, val_lengths = batchConverter(combined_data)
        val_inputs, mask_labels = mask_sequence(val_batch_token, token_to_index['TOKEN_MASK'], val_lengths)
        val_inputs = val_inputs.to(model_engine.device)
        t1 = time.time()
        mask_prediction_logits = model_engine(val_inputs, val_lengths)
        mask_labels = torch.tensor(mask_labels)
        mask_prediction_logits = mask_prediction_logits.to(mask_labels.device)
        mask_prediction_logits = mask_prediction_logits.float()
        mask_prediction_loss = mask_prediction_loss_fn(mask_prediction_logits.view(-1,21), mask_labels.view(-1))
        time_val_ep += time.time() - t1
        mask_prediction_loss = mask_prediction_loss.detach()
        perplexity = calculate_perplexity(mask_prediction_loss).item()
        loss_val_list.append(mask_prediction_loss.item())
        perplexity_list.append(perplexity)
    print('Fold-****val (Ep avg): | Loss = {:.4f} | Perplexity = {:.4f} sec'.format(
                                                                                              f_mean(loss_val_list),
                                                                                              f_mean(perplexity_list)))
    return loss_val_list,perplexity_list,time_val_ep

def data_with_loader(type_='train',batch_size=512, use_distributed=True):
    if type_ != 'train' and type_ != 'val':
        csv_file_path = '../data/pretrain/{}_set'.format(type_)
    elif type_ == 'train':
        csv_file_path = '../data/pretrain/val_data.csv'
    elif type_ == 'val':
        csv_file_path = '../data/pretrain/val_data.csv'
    dataset = AntigenDataset(csv_file_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,drop_last=True)
    return dataset, loader


model = Pretrain()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

model_engine, optimizer, _, _ = deepspeed.initialize(args=args, model=model, optimizer=optimizer, model_parameters=model.parameters())

print('-----Train-----')
print('-----Generate data loader-----')
train_dataset,train_loader = data_with_loader(type_='train',   batch_size=batch_size)
val_dataset,val_loader = data_with_loader(type_='val',batch_size=batch_size)
for epoch in range(num_epochs):
    print('Epoch: {}/{}'.format(epoch + 1, num_epochs))
    print('dir_saver: ', save_dir)
    loss_train_list, perplexity_train_list, time_train_ep = train_step(model_engine, train_loader)
    loss_val_list,perplexity_val_list,time_val_ep = val_step(model_engine,val_loader)
    perplexity_val_mean = f_mean(perplexity_val_list)
    if perplexity_val_mean < perplexity_min:
        perplexity_min, epoch_best = perplexity_val_mean, epoch
        print('****Saving model: Best epoch = {} | perplexity_min = {:.4f}'.format(epoch_best, perplexity_min))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_engine.save_16bit_model(save_dir=save_dir,
                                                   save_filename='pretrained_model{}.bin '.format(epoch),
                                                   exclude_frozen_parameters=False)
    epoch_end_time = time.time()
print('-----Optimization Finished!-----')
