from loguru import logger
import torch
from torch import nn
import lightning as pl
from transformers import Dinov2Model, Dinov2Config

from models.base_model import EEGClassificationModel


class DINO4EEG(EEGClassificationModel):
    def __init__(self,
                 patch_size: int = 1,
                 num_hidden_layers: int = 4,
                 num_attention_heads: int = 8,
                 hidden_act: str = "gelu",
                 dropout: float = 0.1,
                 **kwargs):
        super(DINO4EEG, self).__init__(**kwargs)
        self.save_hyperparameters()

        self.patch_size = patch_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.dropout = dropout

        self.model = Dinov2Model(Dinov2Config(
            num_channels=self.num_channels,
            patch_size=self.patch_size,
            hidden_size=self.h_dim,
            hidden_act=self.hidden_act,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            hidden_dropout_prob=self.dropout,
            attention_probs_dropout_prob=self.dropout,
            reshape_hidden_states=False,
        ))
        self.save_hyperparameters()

    def forward(self, eegs):
        outs = {}
        outs["mel_spec"] = self.mel_spectrogrammer(eegs)  # [b c m t]
        outs["features"] = self.model(
            outs["mel_spec"]).last_hidden_state[:, 0]  # [b p h]
        # outs["logits"] = self.classifier(outs["features"])  # [b l]
        return outs


# Example usage
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_classes = 4
    eeg_num_channels = 32
    model = DINO4EEG(
        eeg_sampling_rate=128, eeg_num_channels=eeg_num_channels,
        num_labels=num_classes,
    ).to(device)
    batch = {
        "eegs": torch.randn([8, eeg_num_channels, 256], device=device),
        "labels": torch.randint(low=0, high=2, size=[8, 4])
    }
    print(model)
    print(model(batch["eegs"]))
    logger.debug(model.step(batch))
    trainer = pl.Trainer(max_epochs=1)
    # trainer.fit(model, train_loader, val_loader)
    # trainer.test(model, test_loader)
