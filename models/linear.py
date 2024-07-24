from loguru import logger
import torch
from torch import nn


from models.base_model import EEGClassificationModel


class Linear4EEG(EEGClassificationModel):
    def __init__(self,
                 **kwargs):
        super(Linear4EEG, self).__init__(**kwargs)

        with torch.no_grad():
            x = torch.randn([1, self.num_channels, self.eeg_samples])
            num_inputs = self.mel_spectrogrammer(x).numel()
            
        # self.classifier = nn.Linear(num_inputs, self.num_labels)
        self.classifier = nn.Linear(num_inputs, self.h_dim)
        self.save_hyperparameters()

    def forward(self, eegs):
        outs = {}
        outs["mel_spec"] = self.mel_spectrogrammer(eegs)  # [b c m t]
        outs["features"] = self.classifier(outs["mel_spec"].flatten(start_dim=1))  # [b l]
        return outs
