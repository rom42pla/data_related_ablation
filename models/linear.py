from loguru import logger
import torch
from torch import nn


from models.base_model import EEGClassificationModel


class Linear4EEG(EEGClassificationModel):
    def __init__(self,
                 **kwargs):
        super(Linear4EEG, self).__init__(**kwargs)
            
        # self.classifier = nn.Linear(num_inputs, self.num_labels)
        self.cls_head = nn.Linear(self.spectrogram_shape.numel(), self.num_labels)
        self.ids_head = nn.Linear(self.spectrogram_shape.numel(), len(self.id2int))
        self.save_hyperparameters()

    def forward(self, mel_spec):
        outs = {}
        spectrogram_flat = mel_spec.flatten(start_dim=1)
        outs["cls_logits"] = self.cls_head(spectrogram_flat)
        outs["ids_logits"] = self.ids_head(spectrogram_flat)
        return outs
