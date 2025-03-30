import argparse
import numpy as np
import pandas as pd
import sys
import os

# 将项目根目录添加到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc, matthews_corrcoef
from torch import optim
import torch
from tqdm import tqdm
import torch.utils.data as Data
from models.tcrLM import Finetune
import torch.nn as nn

f_mean = lambda l: sum(l) / len(l)
threshold = 0.5
tcr_max_len = 34
batch_size = 1024
epochs = 50
vocab = np.load('../data/dict.npy', allow_pickle=True).item()
vocab_size = len(vocab)
args_parser = argparse.ArgumentParser()
args_parser.add_argument('--local_rank', type=int, help='Local rank passed from distributed launcher')
args = args_parser.parse_args()


def transfer(y_prob, threshold=0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])


def performances_to_pd(performances_list):
    metrics_name = ['roc_auc', 'accuracy', 'mcc', 'f1', 'aupr', 'sensitivity', 'specificity', 'precision', 'recall']
    performances_pd = pd.DataFrame(performances_list, columns=metrics_name)
    performances_pd.loc['mean'] = performances_pd.mean(axis=0)
    performances_pd.loc['std'] = performances_pd.std(axis=0)
    return performances_pd


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
    mcc = matthews_corrcoef(y_true, y_pred)
    if print_:
        print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
        print('y_pred: 0 = {} | 1 = {}'.format(Counter(y_pred)[0], Counter(y_pred)[1]))
        print('y_true: 0 = {} | 1 = {}'.format(Counter(y_true)[0], Counter(y_true)[1]))
        print('auc={:.4f}|sensitivity={:.4f}|specificity={:.4f}|acc={:.4f}|mcc={:.4f}'.format(roc_auc, sensitivity,
                                                                                              specificity, accuracy,
                                                                                              mcc))
        print('precision={:.4f}|recall={:.4f}|f1={:.4f}|aupr={:.4f}'.format(precision, recall, f1, aupr))

    return (roc_auc, accuracy, mcc, f1, aupr, sensitivity, specificity, precision, recall)


def make_data(data):
    pep_inputs, tcr_inputs, labels = [], [], []
    pep_lengths, tcr_lengths = [], []
    for pep, tcr, label in zip(data.peptide, data.tcr, data.label):
        pep_lengths.append(len(pep))
        tcr_lengths.append(len(tcr))
        pep, tcr = pep.ljust(tcr_max_len, '-'), tcr.ljust(tcr_max_len, '-')
        pep_input = [[vocab[n] for n in pep]]
        tcr_input = [[vocab[n] for n in tcr]]
        pep_inputs.extend(pep_input)
        tcr_inputs.extend(tcr_input)
        labels.append(label)
    return torch.LongTensor(pep_inputs), torch.LongTensor(tcr_inputs), torch.LongTensor(labels), torch.LongTensor(
        pep_lengths), torch.LongTensor(tcr_lengths)


class MyDataSet(Data.Dataset):
    def __init__(self, pep_inputs, tcr_inputs, labels, pep_length, tcr_length):
        super(MyDataSet, self).__init__()
        self.pep_inputs = pep_inputs
        self.tcr_inputs = tcr_inputs
        self.labels = labels
        self.pep_length = pep_length
        self.tcr_length = tcr_length

    def __len__(self):  # 样本数
        return self.pep_inputs.shape[0]

    def __getitem__(self, idx):
        return self.pep_inputs[idx], self.tcr_inputs[idx], self.labels[idx], self.pep_length[idx], self.tcr_length[idx]


def eval_step(model, val_loader, fold, epoch, epochs, device):
    model.eval()
    torch.manual_seed(66)
    torch.cuda.manual_seed(66)
    with torch.no_grad():
        loss_val_list, dec_attns_val_list = [], []
        y_true_val_list, y_prob_val_list = [], []
        for val_pep_inputs, val_hla_inputs, val_labels, pep_length, tcr_length in tqdm(val_loader, colour='blue'):
            val_pep_inputs = val_pep_inputs.to(device)
            val_hla_inputs = val_hla_inputs.to(device)
            pep_length = pep_length.to(device)
            tcr_length = tcr_length.to(device)
            val_outputs = model(val_pep_inputs, val_hla_inputs, pep_length, tcr_length)
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


def data_with_loader(type_='train', fold=None, batch_size=batch_size):
    if type_ != 'train' and type_ != 'val':
        data = pd.read_csv('../data/finetune/{}_set.csv'.format(type_))
    elif type_ == 'train':
        data = pd.read_csv('../data/finetune/train_fold_{}.csv'.format(fold))
    elif type_ == 'val':
        data = pd.read_csv('../data/finetune/val_fold_{}.csv'.format(fold))

    pep_inputs, hla_inputs, labels, pep_length, tcr_length = make_data(data)
    loader = Data.DataLoader(MyDataSet(pep_inputs, hla_inputs, labels, pep_length, tcr_length), batch_size,
                             shuffle=False, num_workers=0,
                             drop_last=True)
    return data, pep_inputs, hla_inputs, labels, loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

independent_data, independent_pep_inputs, independent_hla_inputs, independent_labels, independent_loader = data_with_loader(
    type_='independent', fold=None, batch_size=batch_size)
triple_data, triple_pep_inputs, triple_hla_inputs, triple_labels, triple_loader = data_with_loader(
    type_='triple', fold=None, batch_size=batch_size)

for n_heads in [1]:
    train_fold_metrics_list, val_fold_metrics_list = [], []
    triple_fold_metrics_list = []
    independent_fold_metrics_list, external_fold_metrics_list, ys_independent_fold_dict, ys_external_fold_dict = [], [], {}, {}
    ep_best = 1
    for fold in range(1,2):
        loss_list = []
        print('=====Fold-{}====='.format(fold))
        print('-----Generate data loader-----')
        train_data, train_pep_inputs, train_hla_inputs, train_labels, train_loader = data_with_loader(type_='train',
                                                                                                      fold=fold,
                                                                                                      batch_size=batch_size)
        val_data, val_pep_inputs, val_hla_inputs, val_labels, val_loader = data_with_loader(type_='val', fold=fold,
                                                                                            batch_size=batch_size)
        print('Fold-{} Label info: Train = {} | Val = {}'.format(fold, Counter(train_data.label),
                                                                 Counter(val_data.label)))
        print('-----Compile model-----')

        model = Finetune().to(device)  # Send model to the device
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        print('-----Load pretrained model-----')
        saved_weights_path = '../pretrained_model/finetuned_model.bin'
        model_weights = torch.load(saved_weights_path)
        model.load_state_dict(model_weights)

        print('-----Evaluate Results-----')
        if ep_best >= 0:
            ys_res_val, loss_res_val_list, metrics_res_val = eval_step(model, val_loader, fold, ep_best, epochs, device)
            ys_res_independent, loss_res_independent_list, metrics_res_independent = eval_step(model,
                                                                                               independent_loader, fold,
                                                                                               ep_best, epochs, device)
            ys_res_triple, loss_res_triple_list, metrics_res_triple = eval_step(model, triple_loader, fold, ep_best,
                                                                                epochs, device)
            ys_res_train, loss_res_train_list, metrics_res_train = eval_step(model, train_loader, fold, ep_best, epochs,
                                                                             device)

            train_fold_metrics_list.append(metrics_res_train)
            val_fold_metrics_list.append(metrics_res_val)
            independent_fold_metrics_list.append(metrics_res_independent)
            triple_fold_metrics_list.append(metrics_res_triple)

print('****Independent set:')
print(performances_to_pd(independent_fold_metrics_list).to_string())
print('****Triple set:')
print(performances_to_pd(triple_fold_metrics_list).to_string())
print('****Train set:')
print(performances_to_pd(train_fold_metrics_list).to_string())
print('****Val set:')
print(performances_to_pd(val_fold_metrics_list).to_string())
