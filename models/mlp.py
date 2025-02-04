from loguru import logger
import torch
from torch import nn


from models.base_model import EEGClassificationModel


class MLP4EEG(EEGClassificationModel):
    def __init__(self,
                 hidden_size=2048,
                 dropout=0.25,
                 **kwargs):
        super(MLP4EEG, self).__init__(**kwargs)

        self.hidden_size = hidden_size
        assert 0 <= dropout < 1
        self.dropout = dropout
            
        # self.classifier = nn.Linear(num_inputs, self.num_labels)
        self.classifier = nn.Sequential(
            nn.Linear(self.spectrogram_shape.numel(), self.hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            
            nn.Linear(self.hidden_size, self.h_dim),
        )
        self.save_hyperparameters()

    def forward(self, wf, mel_spec):
        outs = {}
        outs["features"] = self.classifier(mel_spec.flatten(start_dim=1))  # [b f]
        return outs
