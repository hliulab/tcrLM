import argparse

import os
from collections import Counter
import time

from tcrLM import *
import deepspeed
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, precision_recall_curve, matthews_corrcoef, \
    accuracy_score, precision_score, recall_score, f1_score
import torch.utils.data as Data
from tqdm import tqdm
from encoder import FLASHTransformer

metric_best, ep_best = 0, -1
time_train = 0

threshold = 0.5
args_parser = argparse.ArgumentParser()
args_parser.add_argument('--local_rank', type=int,
                        help='Local rank passed from distributed launcher')
deepspeed.add_config_arguments(args_parser)
args = args_parser.parse_args()
use_cuda = torch.cuda.is_available()
n_heads= 1
n_layers = 1
seq_max_len = 34
d_k = d_v = 64
d_ff = 512
vocab = np.load('/data/ycp/fx/计设/immunity/blosum/dict.npy', allow_pickle=True).item()
vocab_size = len(vocab)
epochs = 35

#构建索引表
residue_tokens = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K',
                  'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'PADDING_MASK']

def transfer(y_prob, threshold=0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])

token_to_index = {}
for i, j in enumerate(residue_tokens):
    token_to_index[j] = i

f_mean = lambda l: sum(l) / len(l)

batch_size = 2048
tcr_max_len = 34
d_model = 512

def calculate_perplexity(loss):
    return torch.exp(loss)

def make_data(data):
    pep_inputs, tcr_inputs, labels = [], [], []
    pep_lengths, tcr_lengths = [], []
    for pep, tcr, label in zip(data.peptide, data.tcr, data.label):
        pep_lengths.append(len(pep))
        tcr_lengths.append(len(tcr))
        pep, tcr = pep.ljust(tcr_max_len, '-'), tcr.ljust(tcr_max_len, '-')
        pep_input = [[vocab[n] for n in pep]]  # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
        tcr_input = [[vocab[n] for n in tcr]]
        pep_inputs.extend(pep_input)
        tcr_inputs.extend(tcr_input)
        labels.append(label)
    return torch.LongTensor(pep_inputs), torch.LongTensor(tcr_inputs), torch.LongTensor(labels),torch.LongTensor(pep_lengths),torch.LongTensor(tcr_lengths)

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup1 = {}
        self.backup2 = {}
    def attack(self, epsilon=1., emb_name='emb'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if emb_name == 'encoder_H.src_emb':
                    self.backup1[name] = param.data.clone()
                if emb_name == 'encoder_P.src_emb':
                    self.backup2[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:

                if emb_name == 'encoder_H.src_emb':
                    assert name in self.backup1
                    param.data = self.backup1[name]
                if emb_name == 'encoder_P.src_emb':
                    assert name in self.backup2
                    param.data = self.backup2[name]
        if emb_name == 'encoder_H.src_emb':
            self.backup1 = {}
        if emb_name == 'encoder_P.src_emb':
            self.backup2 = {}

def performances(y_true, y_pred, y_prob, print_=True):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    try:
        recall = tp / (tp + fn)
    except:
        recall = np.nan

    try:
        precision = tp / (tp + fp)
    except:
        precision = np.nan

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except:
        f1 = np.nan

    roc_auc = roc_auc_score(y_true, y_prob)
    prec, reca, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(reca, prec)
    mcc = matthews_corrcoef(y_true,y_pred)
    if print_:
        print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
        print('y_pred: 0 = {} | 1 = {}'.format(Counter(y_pred)[0], Counter(y_pred)[1]))
        print('y_true: 0 = {} | 1 = {}'.format(Counter(y_true)[0], Counter(y_true)[1]))
        print('auc={:.4f}|sensitivity={:.4f}|specificity={:.4f}|acc={:.4f}|mcc={:.4f}'.format(roc_auc, sensitivity,
                                                                                              specificity, accuracy,mcc
                                                                                              ))
        print('precision={:.4f}|recall={:.4f}|f1={:.4f}|aupr={:.4f}'.format(precision, recall, f1, aupr))

    return (roc_auc, accuracy, mcc, f1, aupr,sensitivity, specificity, precision, recall )


def eval_step(model, val_loader, fold, epoch, epochs, use_cuda=True):
    model.eval()
    torch.manual_seed(66)
    torch.cuda.manual_seed(66)
    with torch.no_grad():
        loss_val_list, dec_attns_val_list = [], []
        y_true_val_list, y_prob_val_list = [], []
        for val_pep_inputs, val_hla_inputs, val_labels,pep_length,tcr_length in tqdm(val_loader,colour='blue'):
            val_outputs = model(val_pep_inputs, val_hla_inputs,pep_length,tcr_length)
            val_labels = val_labels.to(val_outputs.device)
            val_loss = criterion(val_outputs, val_labels)

            y_true_val = val_labels.cpu().numpy()
            y_prob_val = nn.Softmax(dim=1)(val_outputs)[:, 1].cpu().detach().numpy()

            y_true_val_list.extend(y_true_val)
            y_prob_val_list.extend(y_prob_val)
            loss_val_list.append(val_loss)

        y_pred_val_list = transfer(y_prob_val_list, threshold)
        ys_val = (y_true_val_list, y_pred_val_list, y_prob_val_list)

        print('Fold-{} ****Test  Epoch-{}/{}: Loss = {:.6f}'.format(fold, epoch, epochs, f_mean(loss_val_list)))
        metrics_val = performances(y_true_val_list, y_pred_val_list, y_prob_val_list, print_=True)
    return ys_val, f_mean(loss_val_list), metrics_val

def load_prot_flash_small():
    model_data = torch.load("/data/ycp/fx/计设/ProtFlash-main/flash_protein.pt")
    hyper_parameter = model_data["hyper_parameters"]
    model = FLASHTransformer(hyper_parameter['dim'], hyper_parameter['num_tokens'], hyper_parameter['num_layers'], group_size=hyper_parameter['num_tokens'],
                             query_key_dim=hyper_parameter['qk_dim'], max_rel_dist=hyper_parameter['max_rel_dist'], expansion_factor=hyper_parameter['expansion_factor'])
    model.load_state_dict(model_data['state_dict'])
    return model

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

def data_with_loader(type_='train', fold=None, batch_size=batch_size):
    if type_ != 'train' and type_ != 'val':
        data = pd.read_csv('/data/ycp/UnifyImmun/data/data_tcr_new/{}_set_balanced.csv'.format(type_))
    elif type_ == 'train':
        data = pd.read_csv('/data/ycp/UnifyImmun/data/data_tcr_new/train_fold_{}_balanced.csv'.format(fold))
    elif type_ == 'val':
        data = pd.read_csv('/data/ycp/UnifyImmun/data/data_tcr_new/val_fold_{}_balanced.csv'.format(fold))

    pep_inputs, hla_inputs, labels, pep_length,tcr_length = make_data(data)
    loader = Data.DataLoader(MyDataSet(pep_inputs, hla_inputs, labels,pep_length,tcr_length), batch_size, shuffle=False, num_workers=0,pin_memory=True,
                             drop_last=True)
    return data, pep_inputs, hla_inputs, labels, loader

def train_step(model, train_loader, fold, epoch, epochs, use_cuda=True):
    time_train_ep = 0
    start_time = time.time()
    model.train()
    y_true_train_list, y_prob_train_list = [], []
    loss_train_list, dec_attns_train_list = [], []
    fgm = FGM(model)
    for train_pep_inputs, train_hla_inputs, train_labels,pep_length,tcr_length in tqdm(train_loader,colour='yellow'):
        '''
        pep_inputs: [batch_size, pep_len]
        hla_inputs: [batch_size, hla_len]
        train_outputs: [batch_size, 2]
        '''
        optimizer.zero_grad()
        t1 = time.time()
        device = next(model.parameters()).device
        train_pep_inputs = train_pep_inputs.to(device)
        train_pep_inputs = train_pep_inputs.to(device)
        pep_length = pep_length.to(device)
        tcr_length = tcr_length.to(device)
        train_outputs = model(train_pep_inputs, train_hla_inputs,pep_length,tcr_length)
        train_labels = train_labels.to(train_outputs.device)
        train_outputs = train_outputs.float()
        train_loss = criterion(train_outputs, train_labels)
        model.backward(train_loss)
        time_train_ep += time.time() - t1
        fgm.attack(emb_name='encoder_H.src_emb')
        fgm.attack(emb_name='encoder_P.src_emb')
        train_outputs2 = model(train_pep_inputs, train_hla_inputs,pep_length,tcr_length)
        loss_sum = criterion(train_outputs2, train_labels)
        model.backward(loss_sum)
        fgm.restore(emb_name='encoder_H.src_emb')
        fgm.restore(emb_name='encoder_P.src_emb')
        model.step()
        optimizer.zero_grad()
        y_true_train = train_labels.cpu().numpy()
        y_prob_train = nn.Softmax(dim=1)(train_outputs)[:, 1].cpu().detach().numpy()
        y_true_train_list.extend(y_true_train)
        y_prob_train_list.extend(y_prob_train)
        loss_train_list.append(train_loss)
    end_time = time.time()
    total_time = end_time - start_time
    y_pred_train_list = transfer(y_prob_train_list, threshold)
    ys_train = (y_true_train_list, y_pred_train_list, y_prob_train_list)
    print('Fold-{}****Train (Ep avg): Epoch-{}/{} | Loss = {:.4f} | Time = {:.4f} sec'.format(fold, epoch, epochs,
                                                                                              f_mean(loss_train_list),
                                                                                              time_train_ep))
    metrics_train = performances(y_true_train_list, y_pred_train_list, y_prob_train_list, print_=True)
    return ys_train, f_mean(loss_train_list), metrics_train, time_train_ep,total_time

for n_heads in [1]:

    ys_train_fold_dict, ys_val_fold_dict = {}, {}
    train_fold_metrics_list, val_fold_metrics_list = [], []
    independent_fold_metrics_list, external_fold_metrics_list, ys_independent_fold_dict, ys_external_fold_dict = [], [], {}, {}
    triple_fold_metrics_list = []
    attns_train_fold_dict, attns_val_fold_dict, attns_independent_fold_dict, attns_external_fold_dict = {}, {}, {}, {}
    loss_train_fold_dict, loss_val_fold_dict, loss_independent_fold_dict, loss_external_fold_dict = {}, {}, {}, {}
    loss_total_list = []
    for fold in range(1, 6):
        loss_list = []
        print('=====Fold-{}====='.format(fold))
        print('-----Generate data loader-----')
        checkpoint_path = '/data/ycp/fx/计设/model/'
        train_data, train_pep_inputs, train_hla_inputs, train_labels, train_loader = data_with_loader(type_='train',
                                                                                                      fold=fold,
                                                                                                      batch_size=batch_size)
        val_data, val_pep_inputs, val_hla_inputs, val_labels, val_loader = data_with_loader(type_='val', fold=fold,
                                                                                            batch_size=batch_size)
        print('Fold-{} Label info: Train = {} | Val = {}'.format(fold, Counter(train_data.label),
                                                                 Counter(val_data.label)))

        print('-----Compile model-----')
        protflash = load_prot_flash_small()
        pre_model = Pretrain()
        model = Finetune()
        finetune_state_dict = model.state_dict()
        named_parameters = pre_model.named_parameters()
        criterion = nn.CrossEntropyLoss()
        optimizer2 = optim.Adam([
            {'params': pre_model.protflash.parameters(), 'lr': 1e-6},
            {'params': pre_model.projection_mask.parameters()}
        ], lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

        for param in model.encoder_P.parameters():
            param.requires_grad = False

        for param in model.encoder_T.parameters():
            param.requires_grad = False
        params_to_update = {name: param for name, param in model.named_parameters() if param.requires_grad}


        optimizer = optim.Adam(params_to_update.values(), lr=1e-3)
        pre_model, optimizer2, _, _ = deepspeed.initialize(args=args,
                                                             model=pre_model,
                                                             optimizer=optimizer2,
                                                             model_parameters=named_parameters)
        pretrained_state_dict = pre_model.state_dict()
        finetune_state_dict = model.state_dict()

        model, optimizer, _, _ = deepspeed.initialize(args=args,
                                                             model=model,
                                                             optimizer=optimizer,
                                                             model_parameters=model.parameters())
        model.module.load_state_dict(finetune_state_dict)
        i = 0

        print('-----Train-----')
        dir_saver = '/data/ycp/fx/计设/ProtFlash-main/immunity/blosum/test'
        path_saver = '/data/ycp/fx/计设/ProtFlash-main/immunity/blosum/test/随机参数liner_fold{}'.format(fold)

        print('dir_saver: ', dir_saver)
        print('path_saver: ', path_saver)
        metric_best, ep_best = 0, -1
        time_train = 0

        for epoch in range(1, epochs + 1):
            ys_train, loss_train, metrics_train, time_train_ep,train_time = train_step(model, train_loader, fold, epoch,
                                                                                 epochs, use_cuda)  # , dec_attns_train
            ys_val, loss_val_list, metrics_val = eval_step(model, val_loader, fold, epoch, epochs,
                                                           use_cuda)  # , dec_attns_val
            loss_list.append(loss_train)
            metrics_ep_avg = sum(metrics_val[:5]) / 5
            if metrics_ep_avg > metric_best:
                metric_best, ep_best = metrics_ep_avg, epoch
                if not os.path.exists(dir_saver):
                    os.makedirs(dir_saver)
                print('****Saving model: Best epoch = {} | metrics_Best_avg = {:.4f}'.format(ep_best, metric_best))
                print('*****Path saver: ', path_saver)
                model.save_checkpoint(path_saver)
            time_train += time_train_ep
        loss_total_list.append(loss_list)


