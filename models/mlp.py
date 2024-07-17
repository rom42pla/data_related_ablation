from loguru import logger
import torch
from torch import nn


from models.base_model import EEGClassificationModel


class MLP4EEG(EEGClassificationModel):
    def __init__(self,
                 hidden_size=512,
                 **kwargs):
        super(MLP4EEG, self).__init__(**kwargs)

        self.hidden_size = hidden_size
        with torch.no_grad():
            x = torch.randn([1, self.num_channels, self.eeg_samples])
            num_inputs = self.mel_spectrogrammer(x).numel()
            
        # self.classifier = nn.Linear(num_inputs, self.num_labels)
        self.classifier = nn.Sequential(
            nn.Linear(num_inputs, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_labels),
        )
        self.save_hyperparameters()

    def forward(self, eegs):
        outs = {}
        outs["mel_spec"] = self.mel_spectrogrammer(eegs)  # [b c m t]
        outs["logits"] = self.classifier(outs["mel_spec"].flatten(start_dim=1))  # [b l]
        return outs
