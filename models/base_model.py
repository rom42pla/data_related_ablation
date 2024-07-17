from abc import abstractmethod
from loguru import logger
import math
import torch
import torch.nn.functional as F
import lightning as pl
import torchmetrics
from typing import Optional, Union, List
from torchaudio.transforms import MelSpectrogram


class EEGClassificationModel(pl.LightningModule):
    def __init__(self,
                 eeg_sampling_rate: int,
                 eeg_num_channels: int,
                 eeg_samples: int,
                 num_labels: int,
                 eeg_windows_size: Union[int, float] = 0.2,
                 eeg_windows_stride: Union[int, float] = 0.1,
                 n_mels: int = 8,
                 min_freq: Optional[int] = None,
                 max_freq: Optional[int] = None,
                 lr: float = 5e-5):
        super(EEGClassificationModel, self).__init__()
        self.save_hyperparameters()

        # spectrogram params
        self.eeg_sampling_rate = eeg_sampling_rate
        self.eeg_samples = eeg_samples
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
        self.num_labels = num_labels
        self.num_channels = eeg_num_channels
        
        # optimizer params
        assert lr > 0
        self.lr = lr

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    @abstractmethod
    def forward(self, eegs):
        pass

    def step(self, batch):
        phase = "train" if self.training else "val"
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