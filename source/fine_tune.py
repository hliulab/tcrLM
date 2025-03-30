import argparse
import sys
import os

# Add the project's root directory to sys.path to ensure we can import modules from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter
import time
from models.tcrLM import *
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
from models.encoder import FLASHTransformer

# Define the device for training (use GPU if available, else use CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the best metric and best epoch values
metric_best, ep_best = 0, -1
time_train = 0

# Set the threshold for classification
threshold = 0.5
args_parser = argparse.ArgumentParser()
args_parser.add_argument('--local_rank', type=int,
                         help='Local rank passed from distributed launcher')
args_parser.add_argument('--master_port', type=int, default=29500, help='Master port for distributed training')
# Remove deepspeed.add_config_arguments(args_parser)
args = args_parser.parse_args()

# Hyperparameters and model settings
n_heads = 1
n_layers = 1
seq_max_len = 20
d_k = d_v = 64
d_ff = 512
vocab = np.load('../data/dict.npy', allow_pickle=True).item()  # Load the vocabulary for peptides
vocab_size = len(vocab)  # Number of unique tokens in the vocabulary
epochs = 35

# Define the residue tokens (amino acids)
residue_tokens = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K',
                  'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'PADDING_MASK']


def transfer(y_prob, threshold=0.5):
    """
    Converts probabilities into class predictions based on the given threshold.

    Args:
    - y_prob (array-like): The predicted probabilities.
    - threshold (float): The threshold to determine class predictions.

    Returns:
    - A numpy array of class predictions (0 or 1).
    """
    return np.array([[0, 1][x > threshold] for x in y_prob])


# Map each residue token to an index
token_to_index = {}
for i, j in enumerate(residue_tokens):
    token_to_index[j] = i

# Simple function to calculate the mean of a list
f_mean = lambda l: sum(l) / len(l)

# Hyperparameters for batching and TCR processing
batch_size = 2048
tcr_max_len = 34
d_model = 512


# Perplexity calculation from loss (typically used in language models)
def calculate_perplexity(loss):
    """
    Calculates the perplexity from a given loss value.

    Args:
    - loss (torch.Tensor): The loss value.

    Returns:
    - torch.Tensor: The calculated perplexity.
    """
    return torch.exp(loss)


# Function to process data into tensor format
def make_data(data):
    """
    Processes raw data into tensors, padding sequences as needed.

    Args:
    - data (DataFrame): The input data containing peptides, TCRs, and labels.

    Returns:
    - torch.LongTensor: Processed peptide inputs.
    - torch.LongTensor: Processed TCR inputs.
    - torch.LongTensor: Labels.
    - torch.LongTensor: Peptide sequence lengths.
    - torch.LongTensor: TCR sequence lengths.
    """
    pep_inputs, tcr_inputs, labels = [], [], []
    pep_lengths, tcr_lengths = [], []
    for pep, tcr, label in zip(data.peptide, data.tcr, data.label):
        pep_lengths.append(len(pep))
        tcr_lengths.append(len(tcr))
        pep, tcr = pep.ljust(tcr_max_len, '-'), tcr.ljust(tcr_max_len, '-')
        # Convert peptide and TCR to index based on vocabulary
        pep_input = [[vocab[n] for n in pep]]
        tcr_input = [[vocab[n] for n in tcr]]
        pep_inputs.extend(pep_input)
        tcr_inputs.extend(tcr_input)
        labels.append(label)
    # Return processed inputs and labels as tensors
    return torch.LongTensor(pep_inputs), torch.LongTensor(tcr_inputs), torch.LongTensor(labels), torch.LongTensor(
        pep_lengths), torch.LongTensor(tcr_lengths)


# FGM (Fast Gradient Method) for adversarial training
class FGM():
    """
    Class to implement the Fast Gradient Method (FGM) for adversarial training.
    This technique is used to generate adversarial examples by perturbing the model's embeddings.
    """

    def __init__(self, model):
        """
        Initializes the FGM object.

        Args:
        - model (nn.Module): The model to apply adversarial training to.
        """
        self.model = model
        self.backup1 = {}
        self.backup2 = {}

    def attack(self, epsilon=1., emb_name='emb'):
        """
        Applies adversarial perturbation to the embeddings of the model.

        Args:
        - epsilon (float): The magnitude of the adversarial perturbation.
        - emb_name (str): The name of the embedding layer to perturb (either 'encoder_H.src_emb' or 'encoder_P.src_emb').
        """
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
        """
        Restores the model's parameters to their original values after adversarial perturbation.

        Args:
        - emb_name (str): The name of the embedding layer to restore.
        """
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


# Function to calculate performance metrics (accuracy, precision, recall, AUC, etc.)
def performances(y_true, y_pred, y_prob, print_=True):
    """
    Calculates and prints various performance metrics including accuracy, precision, recall, AUC, etc.

    Args:
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.
    - y_prob (array-like): Predicted probabilities.
    - print_ (bool): Whether to print the results.

    Returns:
    - tuple: Calculated metrics (AUC, accuracy, MCC, F1, AUPR, sensitivity, specificity, precision, recall).
    """
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
        # Print detailed metrics
        print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
        print('y_pred: 0 = {} | 1 = {}'.format(Counter(y_pred)[0], Counter(y_pred)[1]))
        print('y_true: 0 = {} | 1 = {}'.format(Counter(y_true)[0], Counter(y_true)[1]))
        print('auc={:.4f}|sensitivity={:.4f}|specificity={:.4f}|acc={:.4f}|mcc={:.4f}'.format(
            roc_auc, sensitivity, specificity, accuracy, mcc))
        print('precision={:.4f}|recall={:.4f}|f1={:.4f}|aupr={:.4f}'.format(precision, recall, f1, aupr))
    return (roc_auc, accuracy, mcc, f1, aupr, sensitivity, specificity, precision, recall)


# Evaluation step for validation set
def eval_step(model, val_loader, fold, epoch, epochs):
    """
    Performs the evaluation step for the validation set.

    Args:
    - model (nn.Module): The model being evaluated.
    - val_loader (DataLoader): The DataLoader for the validation set.
    - fold (int): The current fold number.
    - epoch (int): The current epoch number.
    - epochs (int): The total number of epochs.

    Returns:
    - tuple: True labels, predicted labels, predicted probabilities, and evaluation metrics.
    """
    model.eval()
    torch.manual_seed(66)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(66)
    with torch.no_grad():
        loss_val_list = []
        y_true_val_list, y_prob_val_list = [], []
        for val_pep_inputs, val_hla_inputs, val_labels, pep_length, tcr_length in tqdm(val_loader, colour='blue'):
            # Move data to device
            val_pep_inputs = val_pep_inputs.to(device)
            val_hla_inputs = val_hla_inputs.to(device)
            pep_length = pep_length.to(device)
            tcr_length = tcr_length.to(device)
            val_outputs = model(val_pep_inputs, val_hla_inputs, pep_length, tcr_length)
            val_labels = val_labels.to(device)
            val_loss = criterion(val_outputs, val_labels)

            y_true_val = val_labels.cpu().numpy()
            y_prob_val = nn.Softmax(dim=1)(val_outputs)[:, 1].cpu().detach().numpy()

            y_true_val_list.extend(y_true_val)
            y_prob_val_list.extend(y_prob_val)
            loss_val_list.append(val_loss.item())

        y_pred_val_list = transfer(y_prob_val_list, threshold)
        ys_val = (y_true_val_list, y_pred_val_list, y_prob_val_list)
        print('Fold-{} ****Test  Epoch-{}/{}: Loss = {:.6f}'.format(fold, epoch, epochs, f_mean(loss_val_list)))
        metrics_val = performances(y_true_val_list, y_pred_val_list, y_prob_val_list, print_=True)
    return ys_val, f_mean(loss_val_list), metrics_val


# Function to load training/validation data
def data_with_loader(type_='train', fold=None, batch_size=batch_size):
    """
    Loads the dataset for training or validation and prepares the DataLoader.

    Args:
    - type_ (str): Type of data to load ('train' or 'val').
    - fold (int): The current fold number (for cross-validation).
    - batch_size (int): The batch size for training.

    Returns:
    - tuple: Raw data, processed peptide inputs, processed TCR inputs, labels, and DataLoader.
    """
    if type_ != 'train' and type_ != 'val':
        data = pd.read_csv('../data/finetune/{}_set.csv'.format(type_))
    elif type_ == 'train':
        data = pd.read_csv('../data/finetune/train_fold_{}.csv'.format(fold))
    elif type_ == 'val':
        data = pd.read_csv('../data/finetune/val_fold_{}.csv'.format(fold))

    pep_inputs, hla_inputs, labels, pep_length, tcr_length = make_data(data)
    loader = Data.DataLoader(
        MyDataSet(pep_inputs, hla_inputs, labels, pep_length, tcr_length),
        batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True
    )
    return data, pep_inputs, hla_inputs, labels, loader


# Training step for each epoch
def train_step(model, train_loader, fold, epoch, epochs):
    """
    Executes the training step for each epoch.

    Args:
    - model (nn.Module): The model being trained.
    - train_loader (DataLoader): The DataLoader for the training set.
    - fold (int): The current fold number.
    - epoch (int): The current epoch number.
    - epochs (int): The total number of epochs.

    Returns:
    - tuple: True labels, predicted labels, predicted probabilities, and training metrics.
    """
    time_train_ep = 0
    start_time = time.time()
    model.train()
    y_true_train_list, y_prob_train_list = [], []
    loss_train_list = []
    fgm = FGM(model)
    for train_pep_inputs, train_hla_inputs, train_labels, pep_length, tcr_length in tqdm(train_loader, colour='yellow'):
        # Zero gradients for optimizer
        optimizer.zero_grad()
        t1 = time.time()
        # Move data to device
        train_pep_inputs = train_pep_inputs.to(device)
        train_hla_inputs = train_hla_inputs.to(device)
        pep_length = pep_length.to(device)
        tcr_length = tcr_length.to(device)
        train_labels = train_labels.to(device)
        train_outputs = model(train_pep_inputs, train_hla_inputs, pep_length, tcr_length)
        train_outputs = train_outputs.float()
        train_loss = criterion(train_outputs, train_labels)
        # Forward pass and backpropagation
        train_loss.backward()
        time_train_ep += time.time() - t1

        # Adversarial training: attack
        fgm.attack(emb_name='encoder_H.src_emb')
        fgm.attack(emb_name='encoder_P.src_emb')
        train_outputs2 = model(train_pep_inputs, train_hla_inputs, pep_length, tcr_length)
        loss_sum = criterion(train_outputs2, train_labels)
        loss_sum.backward()
        # Restore model after adversarial attack
        fgm.restore(emb_name='encoder_H.src_emb')
        fgm.restore(emb_name='encoder_P.src_emb')
        optimizer.step()
        optimizer.zero_grad()

        y_true_train = train_labels.cpu().numpy()
        y_prob_train = nn.Softmax(dim=1)(train_outputs)[:, 1].cpu().detach().numpy()
        y_true_train_list.extend(y_true_train)
        y_prob_train_list.extend(y_prob_train)
        loss_train_list.append(train_loss.item())
    end_time = time.time()
    total_time = end_time - start_time
    y_pred_train_list = transfer(y_prob_train_list, threshold)
    ys_train = (y_true_train_list, y_pred_train_list, y_prob_train_list)
    print('Fold-{}****Train (Ep avg): Epoch-{}/{} | Loss = {:.4f} | Time = {:.4f} sec'.format(
        fold, epoch, epochs, f_mean(loss_train_list), time_train_ep))
    metrics_train = performances(y_true_train_list, y_pred_train_list, y_prob_train_list, print_=True)
    return ys_train, f_mean(loss_train_list), metrics_train, time_train_ep, total_time


# Main training loop
for n_heads in [1]:
    ys_train_fold_dict, ys_val_fold_dict = {}, {}
    train_fold_metrics_list, val_fold_metrics_list = [], []
    independent_fold_metrics_list, external_fold_metrics_list, ys_independent_fold_dict, ys_external_fold_dict = [], [], {}, {}
    attns_train_fold_dict, attns_val_fold_dict, attns_independent_fold_dict, attns_external_fold_dict = {}, {}, {}, {}
    loss_train_fold_dict, loss_val_fold_dict, loss_independent_fold_dict, loss_external_fold_dict = {}, {}, {}, {}
    loss_total_list = []
    for fold in range(1, 6):
        loss_list = []
        print('=====Fold-{}====='.format(fold))
        print('-----Generate data loader-----')
        saved_weights_path = '../pretrained_model/pretrained_model_tcr.bin'
        model_weights = torch.load(saved_weights_path)
        train_data, train_pep_inputs, train_hla_inputs, train_labels, train_loader = data_with_loader(
            type_='train', fold=fold, batch_size=batch_size)
        val_data, val_pep_inputs, val_hla_inputs, val_labels, val_loader = data_with_loader(
            type_='val', fold=fold, batch_size=batch_size)
        print('Fold-{} Label info: Train = {} | Val = {}'.format(
            fold, Counter(train_data.label), Counter(val_data.label)))

        print('-----Compile model-----')
        model = Finetune()
        criterion = nn.CrossEntropyLoss()

        # Freeze the encoder weights
        for param in model.encoder_P.parameters():
            param.requires_grad = False
        for param in model.encoder_T.parameters():
            param.requires_grad = False

        # Only update parameters that require gradients
        params_to_update = {name: param for name, param in model.named_parameters() if param.requires_grad}
        optimizer = optim.Adam(params_to_update.values(), lr=1e-3)

        # Move model to device
        model = model.to(device)
        model_weights.pop("fc_out.weight", None)
        model_weights.pop("fc_out.bias", None)
        model.encoder_T.load_state_dict(model_weights)
        model.encoder_P.load_state_dict(model_weights)
        print('-----Train-----')
        dir_saver = '../finetuned_tcrLM/'
        path_saver = '../finetuned_tcrLM/finetuned_model{}'.format(fold)

        print('dir_saver: ', dir_saver)
        print('path_saver: ', path_saver)
        metric_best, ep_best = 0, -1
        time_train = 0

        for epoch in range(1, epochs + 1):
            ys_train, loss_train, metrics_train, time_train_ep, train_time = train_step(
                model, train_loader, fold, epoch, epochs)
            ys_val, loss_val_list, metrics_val = eval_step(
                model, val_loader, fold, epoch, epochs)
            loss_list.append(loss_train)
            metrics_ep_avg = sum(metrics_val[:5]) / 5
            if metrics_ep_avg > metric_best:
                metric_best, ep_best = metrics_ep_avg, epoch
                if not os.path.exists(path_saver):
                    os.makedirs(path_saver)
                print('****Saving model: Best epoch = {} | metrics_Best_avg = {:.4f}'.format(ep_best, metric_best))
                print('*****Path saver: ', path_saver)
                # Save the model weights
                torch.save(model.state_dict(), os.path.join(path_saver, f'finetuned_model{epoch}.bin'))
            time_train += time_train_ep
        loss_total_list.append(loss_list)
