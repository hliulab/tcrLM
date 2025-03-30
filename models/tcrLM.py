import torch.nn as nn
from models.encoder import *
threshold = 0.5

class Pretrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.protflash = FLASHTransformer(dim=512, num_tokens=21, depth=48, group_size=21,
                                          query_key_dim=128, max_rel_dist=32, expansion_factor=2)

        self.fc_out = nn.Linear(512, 20)
    def forward(self, x, lengths):
        outputs = self.protflash(x, lengths)
        mask_prediction_logits = self.fc_out(outputs)
        return  mask_prediction_logits


class Finetune(nn.Module):
    def __init__(self, seq_max_len=34, feature_size=512):
        super().__init__()
        self.encoder_T = FLASHTransformer(dim=512, num_tokens=21, depth=48, group_size=21,
                                          query_key_dim=128, max_rel_dist=32, expansion_factor=2)

        self.encoder_P=  FLASHTransformer(dim=512, num_tokens=21, depth=48, group_size=21,
                                          query_key_dim=128, max_rel_dist=32, expansion_factor=2)
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

class Finetune_drop_encoder(nn.Module):
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
            # output layer
            nn.Linear(64, 2)
        )

    def forward(self, combined_outputs):
        logits = self.projection(combined_outputs)
        return logits

class Finetune_one_liner(nn.Module):
    def __init__(self, seq_max_len=34, feature_size=512):
        super().__init__()
        self.seq_max_len = seq_max_len
        self.feature_size = feature_size
        self.fc_out = nn.Linear((34+34) * self.feature_size, 2)


    def forward(self, combined_outputs):
        logits = self.fc_out(combined_outputs)
        return logits
