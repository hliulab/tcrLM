import argparse
import os
from collections import Counter
import time
import sys
# 将项目根目录添加到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.tcrLM import *
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
from utils import *
from models.tcr_encoder import FLASHTransformer
# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
metric_best, ep_best = 0, -1
time_train = 0
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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
vocab = np.load('../data/dict.npy', allow_pickle=True).item()
vocab_size = len(vocab)
epochs = 20

#构建索引表
residue_tokens = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K',
                  'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'PADDING_MASK']

def transfer(y_prob, threshold=0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])

token_to_index = {}
for i, j in enumerate(residue_tokens):
    token_to_index[j] = i

f_mean = lambda l: sum(l) / len(l)

batch_size = 512
tcr_max_len = 34
pep_max_len = 34
d_model = 512

def calculate_perplexity(loss):
    return torch.exp(loss)

def make_data(data):
    pep_inputs, tcr_inputs, labels = [], [], []
    pep_lengths, tcr_lengths = [], []
    for pep, tcr, label in zip(data.peptide, data.tcr, data.label):
        pep_lengths.append(len(pep))
        tcr_lengths.append(len(tcr))
        pep, tcr = pep.ljust(tcr_max_len, '-'), tcr.ljust(pep_max_len, '-')
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

def data_with_loader(type_='train', fold=None, batch_size=batch_size):
    if type_ != 'train' and type_ != 'val':
        data = pd.read_csv('../data/finetune/{}_set.csv'.format(type_))
    elif type_ == 'train':
        data = pd.read_csv('../data/finetune/train_fold_{}.csv'.format(fold))
    elif type_ == 'val':
        data = pd.read_csv('../data/finetune/val_fold_{}.csv'.format(fold))
    pep_inputs, hla_inputs, labels, pep_length,tcr_length = make_data(data)
    loader = Data.DataLoader(MyDataSet(pep_inputs, hla_inputs, labels,pep_length,tcr_length), batch_size, shuffle=False, num_workers=0,pin_memory=True,
                             drop_last=True)
    return data, pep_inputs, hla_inputs, labels, loader

def train_step(pep_embedding,tcr_embedding,model, train_loader, fold, epoch, epochs):
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
        train_hla_inputs = train_hla_inputs.to(device)
        pep_length = pep_length.to(device)
        tcr_length = tcr_length.to(device)
        with torch.no_grad():
            pep_output = pep_embedding(train_pep_inputs,pep_length)
            tcr_output = tcr_embedding(train_hla_inputs,tcr_length)
            total_output = torch.cat((pep_output, tcr_output), dim=1)
            total_output = total_output.view(total_output.shape[0], -1)
            total_output = total_output.half()
        train_outputs = model(total_output)
        train_labels = train_labels.to(train_outputs.device)
        train_outputs = train_outputs.float()
        train_loss = criterion(train_outputs, train_labels)
        model.backward(train_loss, retain_graph=True)
        time_train_ep += time.time() - t1
        fgm.attack(emb_name='encoder_H.src_emb')
        fgm.attack(emb_name='encoder_P.src_emb')
        train_outputs2 = model(total_output)
        loss_sum = criterion(train_outputs2, train_labels)
        model.backward(loss_sum)
        fgm.restore(emb_name='encoder_H.src_emb')
        fgm.restore(emb_name='encoder_P.src_emb')
        model.step()
        optimizer.zero_grad()
        train_loss = train_loss.detach()
        loss_sum = loss_sum.detach()
        y_true_train = train_labels.detach().cpu().numpy()
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

def eval_step(pep_embedding,tcr_embedding,model, val_loader, fold, epoch, epochs):
    model.eval()
    torch.manual_seed(66)
    torch.cuda.manual_seed(66)
    with torch.no_grad():
        loss_val_list, dec_attns_val_list = [], []
        y_true_val_list, y_prob_val_list = [], []

        for val_pep_inputs, val_hla_inputs, val_labels,pep_length,tcr_length in tqdm(val_loader,colour='blue'):
            device = next(model.parameters()).device
            val_pep_inputs = val_pep_inputs.to(device)
            val_hla_inputs = val_hla_inputs.to(device)
            pep_length = pep_length.to(device)
            tcr_length = tcr_length.to(device)
            pep_output = pep_embedding(val_pep_inputs,pep_length)
            tcr_output = tcr_embedding(val_hla_inputs,tcr_length)
            total_output = torch.cat((pep_output, tcr_output), dim=1)
            total_output = total_output.view(total_output.shape[0], -1)
            total_output = total_output.half()
            val_outputs = model(total_output)
            val_labels = val_labels.to(val_outputs.device)
            val_loss = criterion(val_outputs, val_labels)
            y_true_val = val_labels.cpu().numpy()
            y_prob_val = nn.Softmax(dim=1)(val_outputs)[:, 1].cpu().detach().numpy()

            y_true_val_list.extend(y_true_val)
            y_prob_val_list.extend(y_prob_val)
            loss_val_list.append(val_loss)
        y_pred_transfer_val_list = transfer(y_prob_val_list, threshold)
        ys_val = (y_true_val_list, y_pred_transfer_val_list, y_prob_val_list)

        print('Fold-{} ****Test  Epoch-{}/{}: Loss = {:.6f}'.format(fold, epoch, epochs, f_mean(loss_val_list)))
        metrics_val = performances(y_true_val_list, y_pred_transfer_val_list, y_prob_val_list, print_=True)
    return ys_val, f_mean(loss_val_list), metrics_val

if __name__ == '__main__':
    for n_heads in [1]:
        ys_train_fold_dict, ys_val_fold_dict = {}, {}
        train_fold_metrics_list, val_fold_metrics_list = [], []
        covid_fold_metrics_list = []
        independent_fold_metrics_list, external_fold_metrics_list, ys_independent_fold_dict, ys_external_fold_dict = [], [], {}, {}
        triple_fold_metrics_list = []
        attns_train_fold_dict, attns_val_fold_dict, attns_independent_fold_dict, attns_external_fold_dict = {}, {}, {}, {}
        loss_train_fold_dict, loss_val_fold_dict, loss_independent_fold_dict, loss_external_fold_dict = {}, {}, {}, {}
        loss_total_list = []
        for fold in range(1, 6):
            loss_list = []
            print('=====Fold-{}====='.format(fold))
            print('-----Generate data loader-----')
            train_data, train_pep_inputs, train_hla_inputs, train_labels, train_loader = data_with_loader(type_='train',
                                                                                                          fold=fold,
                                                                                                          batch_size=batch_size)
            val_data, val_pep_inputs, val_hla_inputs, val_labels, val_loader = data_with_loader(type_='val', fold=fold,
                                                                                                batch_size=batch_size)
            independent_data, independent_pep_inputs, independent_hla_inputs, independent_labels, independent_loader = data_with_loader(
                type_='independent', fold=None, batch_size=batch_size)

            triple_data, triple_pep_inputs, triple_hla_inputs, triple_labels, triple_loader = data_with_loader(
                type_='triple', fold=None, batch_size=batch_size)
            covid_data, covid_pep_inputs, covid_hla_inputs, covid_labels, covid_loader = data_with_loader(
                type_='covid', fold=None, batch_size=batch_size)

            print('Fold-{} Label info: Train = {} | Val = {}'.format(fold, Counter(train_data.label),
                                                                     Counter(val_data.label)))

            print('-----Compile model-----')

            model = Finetune()
            criterion = nn.CrossEntropyLoss()

            params_to_update = {name: param for name, param in model.named_parameters() if param.requires_grad}

            optimizer = optim.Adam(params_to_update.values(), lr=1e-4)

            model, optimizer, _, _ = deepspeed.initialize(args=args,
                                                                 model=model,
                                                                 optimizer=optimizer,
                                                                 model_parameters=model.parameters())
            print('-----Train-----')
            dir_saver = '../finetune'
            path_saver = '../finetune/finetune_model{}'.format(fold)
            tcr_file_path = '../pretrained_model/tcr_pretrained_model.bin'
            pep_file_path = '../pretrained_model/pep_pretrained_model.bin'
            tcr_weights = torch.load(tcr_file_path)
            pep_weights = torch.load(pep_file_path)
            loss_train_list = []
            loss_val_list = []
            print('dir_saver: ', dir_saver)
            print('path_saver: ', path_saver)
            metric_best, ep_best = 0, -1
            time_train = 0
            encoder_T = FLASHTransformer(dim=512, num_tokens=21, depth=48, group_size=21,
                                         query_key_dim=128, max_rel_dist=32, expansion_factor=2)
            encoder_T.load_state_dict(tcr_weights)

            encoder_P = pep_encoder(dim=512, num_tokens=21, depth=48, group_size=21,
                                  query_key_dim=128, max_rel_dist=32, expansion_factor=2)
            state_dict = encoder_P.state_dict()
            pep_state_dict = {}
            for name, param in pep_weights.items():
                # print(name)
                new_name = name.replace('encoder_P.', '')
                if new_name in state_dict:
                    pep_state_dict[new_name] = param
            encoder_P.load_state_dict(pep_state_dict)
            # 将 embedding 层移动到 model 的设备
            model_device = next(model.parameters()).device
            encoder_T = encoder_T.to(model_device)
            encoder_P = encoder_P.to(model_device)
            for epoch in range(1, epochs + 1):

                ys_train, loss_train, metrics_train, time_train_ep,train_time = train_step(encoder_P,encoder_T,model, train_loader, fold, epoch,
                                                                                     epochs)  # , dec_attns_train
                ys_val, loss_val, metrics_val = eval_step(encoder_P,encoder_T,model, val_loader, fold, epoch, epochs)  # , dec_attns_val
                loss_train_list.append(loss_train)
                loss_val_list.append(loss_val)
                metrics_ep_avg = sum(metrics_val[:5]) / 5
                if metrics_ep_avg > metric_best:
                    metric_best, ep_best = metrics_ep_avg, epoch
                    if not os.path.exists(dir_saver):
                        os.makedirs(dir_saver)
                    print('****Saving model: Best epoch = {} | metrics_Best_avg = {:.4f}'.format(ep_best, metric_best))
                    print('*****Path saver: ', path_saver)
                    model.save_16bit_model(save_dir=path_saver,
                                           save_filename='finetuned_model.bin',
                                           exclude_frozen_parameters=False)
                time_train += time_train_ep

            loss_df = pd.DataFrame({
                'train_loss': [tensor.cpu().numpy() for tensor in loss_train_list],
                'val_loss': [tensor.cpu().numpy() for tensor in loss_val_list]
            })
            loss_df.to_csv('loss_history.csv', index=False)
            print('-----Optimization Finished!-----')
            print('-----Evaluate Results-----')
            if ep_best >= 0:
                print('*****Path saver: ', path_saver)
                state_dict = torch.load(path_saver + "/finetuned_model.bin")
                new_state_dict = {}
                for name, param in state_dict.items():
                    new_name = "module." + name
                    new_state_dict[new_name] = param
                model.load_state_dict(new_state_dict)

                ys_res_train, loss_res_train_list, metrics_res_train = eval_step(encoder_P,encoder_T, model, train_loader, fold,
                                                                                 ep_best,
                                                                                 epochs)  # , train_res_attns
                ys_res_val, loss_res_val_list, metrics_res_val = eval_step(encoder_P,encoder_T,model, val_loader, fold, ep_best,
                                                                           epochs)  # , val_res_attns
                independent_result, loss_res_independent_list, metrics_res_independent = eval_step(encoder_P,encoder_T,model,
                                                                                                   independent_loader,
                                                                                                   fold,
                                                                                                   ep_best, epochs)  # , independent_res_attns
                external_result, loss_res_covid_list, metrics_res_covid = eval_step(encoder_P,encoder_T,model, covid_loader, fold,
                                                                                          ep_best, epochs)  # , external_res_attns

                triple_result, loss_res_triple_list, metrics_res_triple = eval_step(encoder_P,encoder_T,model, triple_loader, fold,
                                                                                    ep_best, epochs)  # , triple_res_attns
                independent_result_data_dict = {}
                covid_result_data_dict = {}
                triple_result_data_dict = {}

                for i in range(len(independent_result[0])):
                    y_true = independent_result[0][i].item()
                    y_pred = independent_result[1][i].item()
                    y_prob = independent_result[2][i].item()

                    independent_result_data_dict[i] = {
                        'y_true': y_true,
                        'y_pred': y_pred,
                        'y_prob': y_prob
                    }

                for i in range(len(external_result[0])):
                    y_true = external_result[0][i].item()
                    y_pred = external_result[1][i].item()
                    y_prob = external_result[2][i].item()

                    covid_result_data_dict[i] = {
                        'y_true': y_true,
                        'y_pred': y_pred,
                        'y_prob': y_prob
                    }

                for i in range(len(triple_result[0])):
                    y_true = triple_result[0][i].item()
                    y_pred = triple_result[1][i].item()
                    y_prob = triple_result[2][i].item()

                    triple_result_data_dict[i] = {
                        'y_true': y_true,
                        'y_pred': y_pred,
                        'y_prob': y_prob
                    }
                # Convert the result_data_dict values to a list
                independent_result_data = list(independent_result_data_dict.values())
                covid_result_data = list(covid_result_data_dict.values())
                triple_result_data = list(triple_result_data_dict.values())

                # Create a DataFrame from the result_data
                independent_result_df = pd.DataFrame(independent_result_data)
                covid_result_df = pd.DataFrame(covid_result_data)
                triple_result_df = pd.DataFrame(triple_result_data)
                # Save the DataFrame to a CSV file
                independent_result_df.to_csv('independent_result_data{}.csv'.format(fold), index=False)
                covid_result_df.to_csv('covid_result_data{}.csv'.format(fold), index=False)
                triple_result_df.to_csv('triple_result_data{}.csv'.format(fold), index=False)
                train_fold_metrics_list.append(metrics_res_train)
                val_fold_metrics_list.append(metrics_res_val)
                independent_fold_metrics_list.append(metrics_res_independent)
                covid_fold_metrics_list.append(metrics_res_covid)
                triple_fold_metrics_list.append(metrics_res_triple)

    print('****Independent set:')
    print(performances_to_pd(independent_fold_metrics_list).to_string())
    print('****External set:')
    print(performances_to_pd(covid_fold_metrics_list).to_string())
    print('****Triple set:')
    print(performances_to_pd(triple_fold_metrics_list).to_string())
    print('****Train set:')
    print(performances_to_pd(train_fold_metrics_list).to_string())
    print('****Val set:')
    print(performances_to_pd(val_fold_metrics_list).to_string())

