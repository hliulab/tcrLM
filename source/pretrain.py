import argparse
import sys
import os

# Add the project's root directory to sys.path to ensure we can import modules from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import *
from models.tcrLM import *
from torch.utils.data import DataLoader  # 确保 DataLoader 被正确导入

# 定义 device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

batch_size = 128
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
args_parser.add_argument('--local_rank', type=int, help='Local rank passed from distributed launcher')
args_parser.add_argument('--master_port', type=int, default=29500, help='Master port for distributed training')
args = args_parser.parse_args()
num_epochs = 2

def train_step(model, optimizer, train_loader):
    """
    This function executes a single training step for a model, including forward pass,
    loss calculation, backward pass, optimizer step, and tracking metrics like loss
    and perplexity. It also saves model checkpoints periodically during training.
    """
    time_train_ep = 0
    loss_train_list = []
    perplexity_list = []
    model.train()
    step = 0
    for row_batch in tqdm(train_loader, colour='yellow'):
        sequence_ids, sequences = row_batch
        combined_data = [(id_, seq) for id_, seq in zip(sequence_ids, sequences)]
        optimizer.zero_grad(set_to_none=True)
        ids, train_batch_token, train_lengths = batchConverter(combined_data)
        train_inputs, mask_labels = mask_sequence(train_batch_token, token_to_index['TOKEN_MASK'], train_lengths)
        t1 = time.time()
        # 将输入移动到 device 上
        train_inputs = train_inputs.to(device)
        mask_prediction_logits = model(train_inputs, train_lengths)
        # 直接在创建时指定 device
        mask_labels = torch.tensor(mask_labels, device=device)
        mask_prediction_logits = mask_prediction_logits.float()
        mask_prediction_loss = mask_prediction_loss_fn(mask_prediction_logits.view(-1, 20), mask_labels.view(-1))
        time_train_ep += time.time() - t1
        mask_prediction_loss.backward()
        optimizer.step()
        mask_prediction_loss = mask_prediction_loss.detach()
        perplexity = calculate_perplexity(mask_prediction_loss).item()
        loss_train_list.append(mask_prediction_loss.item())
        perplexity_list.append(perplexity)
        print('Fold-****Train (Ep avg): | Loss = {:.4f} | Perplexity = {:.4f}'.format(mask_prediction_loss, perplexity))
        # Save checkpoint 每200步保存一次
        if step % 200 == 0:
            ckpt_id = mask_prediction_loss.item()
            if not os.path.exists(checkpoint_save_dir):
                os.makedirs(checkpoint_save_dir)
            torch.save(model.state_dict(), os.path.join(checkpoint_save_dir, f"model_checkpoint_{ckpt_id}.pt"))
        step += 1
    return loss_train_list, perplexity_list, time_train_ep

def val_step(model, val_loader):
    """
    This function performs a validation step for the model. It calculates the loss and
    perplexity for each batch in the validation dataset, and tracks the total time taken
    for the validation epoch.
    """
    time_val_ep = 0
    loss_val_list = []
    perplexity_list = []
    model.eval()
    for row_batch in tqdm(val_loader, colour='yellow'):
        sequence_ids, sequences, classification_labels = row_batch
        combined_data = [(id_, seq) for id_, seq in zip(sequence_ids, sequences)]
        ids, val_batch_token, val_lengths = batchConverter(combined_data)
        val_inputs, mask_labels = mask_sequence(val_batch_token, token_to_index['TOKEN_MASK'], val_lengths)
        # 将输入移动到 device 上
        val_inputs = val_inputs.to(device)
        t1 = time.time()
        mask_prediction_logits = model(val_inputs, val_lengths)
        mask_labels = torch.tensor(mask_labels, device=device)
        mask_prediction_logits = mask_prediction_logits.float()
        # 注意：这里 view(-1, 21) 的维度要与模型输出对应
        mask_prediction_loss = mask_prediction_loss_fn(mask_prediction_logits.view(-1, 21), mask_labels.view(-1))
        time_val_ep += time.time() - t1
        mask_prediction_loss = mask_prediction_loss.detach()
        perplexity = calculate_perplexity(mask_prediction_loss).item()
        loss_val_list.append(mask_prediction_loss.item())
        perplexity_list.append(perplexity)
    print('Fold-****val (Ep avg): | Loss = {:.4f} | Perplexity = {:.4f}'.format(
          f_mean(loss_val_list), f_mean(perplexity_list)))
    return loss_val_list, perplexity_list, time_val_ep


def data_with_loader(type_='train', batch_size=512):
    if type_ != 'train' and type_ != 'val':
        csv_file_path = '../data/pretrain/{}_set'.format(type_)
    elif type_ == 'train':
        csv_file_path = '../data/pretrain/train_data.csv'
    elif type_ == 'val':
        csv_file_path = '../data/pretrain/val_data.csv'
    dataset = AntigenDataset(csv_file_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return dataset, loader

# Initialize model and optimizer
model = Pretrain().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

print('-----Train-----')
print('-----Generate data loader-----')
train_dataset, train_loader = data_with_loader(type_='train', batch_size=batch_size)
val_dataset, val_loader = data_with_loader(type_='val', batch_size=batch_size)

for epoch in range(num_epochs):
    print('Epoch: {}/{}'.format(epoch + 1, num_epochs))
    print('dir_saver: ', save_dir)
    loss_train_list, perplexity_train_list, time_train_ep = train_step(model, optimizer, train_loader)
    loss_val_list, perplexity_val_list, time_val_ep = val_step(model, val_loader)
    perplexity_val_mean = f_mean(perplexity_val_list)
    if perplexity_val_mean < perplexity_min:
        perplexity_min, epoch_best = perplexity_val_mean, epoch
        print('****Saving model: Best epoch = {} | perplexity_min = {:.4f}'.format(epoch_best, perplexity_min))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model.state_dict(), os.path.join(save_dir, f'pretrained_model_{epoch}.bin'))

print('-----Optimization Finished!-----')
