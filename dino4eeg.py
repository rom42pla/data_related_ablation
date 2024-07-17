from loguru import logger
import math
import torch
from torch import nn
import torch.nn.functional as F
import lightning as pl
import torchmetrics
from transformers import AutoImageProcessor, Dinov2Model, Dinov2Config
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Union, List
from torchaudio.transforms import MelSpectrogram
import wandb

# Placeholder dataset class for EEG data


# class EEGDataset(Dataset):
#     def __init__(self, data, labels, image_processor):
#         self.data = data
#         self.labels = labels
#         self.image_processor = image_processor

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         image = self.data[idx]
#         label = self.labels[idx]
#         inputs = self.image_processor(image, return_tensors="pt")
#         return inputs['pixel_values'].squeeze(0), torch.tensor(label)

# Lightning module for EEG classification using DINOv2


class DINO4EEG(pl.LightningModule):
    def __init__(self,
                 eeg_sampling_rate: int,
                 eeg_num_channels: int,
                 num_labels: int,
                 eeg_windows_size: Union[int, float] = 0.2,
                 eeg_windows_stride: Union[int, float] = 0.1,
                 n_mels: int = 8,
                 min_freq: Optional[int] = None,
                 max_freq: Optional[int] = None,


                 patch_size: int = 1,
                 hidden_size: int = 512,
                 num_hidden_layers: int = 4,
                 num_attention_heads: int = 8,
                 hidden_act: str = "gelu",
                 dropout: float = 0.1,
                 lr: float = 5e-5):
        super(DINO4EEG, self).__init__()
        self.save_hyperparameters()

        # spectrogram params
        self.eeg_sampling_rate = eeg_sampling_rate
        self.n_mels = n_mels
        if min_freq is not None:
            assert min_freq >= 0
        self.min_freq = min_freq
        if max_freq is not None:
            assert max_freq >= 0
        self.max_freq = max_freq
        assert eeg_windows_size > 0
        self.eeg_windows_size: int = math.floor(
            eeg_windows_size * self.eeg_sampling_rate)
        assert eeg_windows_stride > 0
        self.eeg_windows_stride: int = math.floor(
            eeg_windows_stride * self.eeg_sampling_rate)
        self.mel_spectrogrammer = MelSpectrogram(
            sample_rate=self.eeg_sampling_rate,
            f_min=self.min_freq,
            f_max=self.max_freq,
            n_mels=self.n_mels,
            n_fft=max(128, self.eeg_windows_size),
            power=1,
            win_length=self.eeg_windows_size,
            hop_length=self.eeg_windows_stride,
            pad=math.ceil(self.eeg_windows_stride//2),
            center=True,
            normalized=True,
        ).float()

        # model params
        self.num_labels = num_labels
        self.patch_size = patch_size
        self.num_channels = eeg_num_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.dropout = dropout

        # self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Dinov2Model(Dinov2Config(
            num_channels=self.num_channels,
            patch_size=self.patch_size,
            hidden_size=self.hidden_size,
            hidden_act=self.hidden_act,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            hidden_dropout_prob=self.dropout,
            attention_probs_dropout_prob=self.dropout,
            reshape_hidden_states=False,
        ))
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

        # optimizer params
        assert lr > 0
        self.lr = lr
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, eegs):
        outs = {}
        outs["mel_spec"] = self.mel_spectrogrammer(eegs)  # [b c m t]
        outs["features"] = self.model(
            outs["mel_spec"]).last_hidden_state  # [b p h]
        outs["logits"] = self.classifier(outs["features"][:, 0])  # [b l]
        return outs

    def step(self, batch):
        phase = "train" if self.training else "val"
        # defines the metrics
        if self.trainer.global_step == 0:
            for metric_name, direction in [
                ("accuracy", "max"),
                ("loss", "min"),
            ]:
                wandb.define_metric(
                    f"{metric_name}/{phase}", summary=direction)
        outs = {
            "labels": batch["labels"].float().to(self.device),
            **self(batch["eegs"].float().to(self.device)),
        }
        outs["loss"] = F.binary_cross_entropy_with_logits(
            input=outs["logits"], target=outs["labels"])
        outs["metrics"] = {
            "accuracy": torchmetrics.functional.accuracy(preds=outs["logits"], target=outs["labels"], task="multilabel", num_labels=self.num_labels, average="micro"),
            "loss": outs["loss"],
        }
        for metric_name, metric_value in outs["metrics"].items():
            self.log(name=f"{metric_name}/{phase}", value=metric_value, prog_bar=True,
                     on_step=False, on_epoch=True, batch_size=batch["eegs"].shape[0])
        return outs

    def training_step(self, batch, batch_idx):
        outs = self.step(batch)
        return outs

    def validation_step(self, batch, batch_idx):
        outs = self.step(batch)
        return outs

    def test_step(self, batch, batch_idx):
        outs = self.step(batch)
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
