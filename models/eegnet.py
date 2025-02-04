from loguru import logger
import torch
from torch import nn


from models.base_model import EEGClassificationModel


class EEGNet(EEGClassificationModel):
    def __init__(self, F1=8, D=2, **kwargs):
        super(EEGNet, self).__init__(**kwargs)

        self.F1 = F1
        self.D = D
        self.F2 = F1 * self.D

        self.net = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.eeg_sampling_rate//2), padding="same", bias=False),
            nn.BatchNorm2d(self.F1),
            nn.Conv2d(self.F1, self.D * self.F1, (self.num_channels, 1), groups=self.F1, padding="valid", bias=False),
            nn.BatchNorm2d(self.D * self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25),

            nn.Conv2d(self.D * self.F1, self.F2, (1, 16), padding="same", bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25),
            nn.Flatten(start_dim=1),
        )
        self.h_dim = self.F2 * (self.eeg_samples // 32) 

        self.cls_head = nn.Linear(self.h_dim, self.num_labels)
        if self.predict_ids:
            self.ids_head = nn.Linear(self.h_dim, len(self.id2int))
        self.save_hyperparameters()

    def forward(self, wf, **kwargs):
        outs = {}

        # feature extraction
        features = self.net(wf.unsqueeze(1))
        assert len(features.shape) == 2 and features.shape[1] == self.h_dim

        # classification
        outs["cls_logits"] = self.cls_head(features)
        if self.predict_ids:
            outs["ids_logits"] = self.ids_head(features)
        return outs
