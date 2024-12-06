import numpy as np
import pandas as pd
import torch.nn as nn
from models.tcr_encoder import *
from torch.cuda.amp import autocast
from models.pep_encoder import pep_encoder

threshold = 0.5
seq_max_len = 20
model_data = torch.load('../pretrained_model/flash_protein.pt')

class Pretrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FLASHTransformer(dim=512, num_tokens=21, depth=48, group_size=21,
                                          query_key_dim=128, max_rel_dist=32, expansion_factor=2)
        self.fc_out = nn.Linear(512, 20)
    def forward(self, x, lengths):
        outputs = self.encoder(x, lengths)
        mask_prediction_logits = self.fc_out(outputs)
        return  mask_prediction_logits

class Finetune(nn.Module):
    def __init__(self, seq_max_len=34, feature_size=512):
        super().__init__()
        self.seq_max_len = seq_max_len
        self.feature_size = feature_size
        self.projection = nn.Sequential(
            nn.Linear((34+34) * self.feature_size, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 2)
        )

    def forward(self, combined_outputs):
        logits = self.projection(combined_outputs)
        return logits

