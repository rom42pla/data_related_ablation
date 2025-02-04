from abc import abstractmethod
from colorama import Fore
from loguru import logger
import math
import torch
import torch.nn.functional as F
import lightning as pl
import torchaudio
import torchmetrics
from typing import Optional, Union, List
from torch import nn
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
                 h_dim: int = 512,
                 predict_ids: bool = True,
                 ids: Optional[List[str]] = None,
                 lr: float = 5e-5):
        super(EEGClassificationModel, self).__init__()
        self.save_hyperparameters()

        # spectrogram params
        self.eeg_sampling_rate = eeg_sampling_rate
        self.nyquist_freq = self.eeg_sampling_rate // 2
        self.eeg_samples = eeg_samples
        self.n_mels = n_mels
        if min_freq is not None:
            assert min_freq >= 0
        else:
            min_freq = 0
        self.min_freq = min_freq
        if max_freq is not None:
            assert max_freq >= 0
        else:
            max_freq = 100
        self.max_freq = min(self.nyquist_freq, max_freq)
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
        with torch.no_grad():
            x = torch.randn([1, self.num_channels, self.eeg_samples])
            self.spectrogram_shape = self.mel_spectrogrammer(x).shape

        # heads params
        assert isinstance(h_dim, int) and h_dim > 0, h_dim
        self.h_dim = h_dim
        self.cls_head = nn.Linear(self.h_dim, self.num_labels)
        assert isinstance(predict_ids, bool)
        self.predict_ids = predict_ids
        if self.predict_ids:
            assert ids is not None and len(ids) > 1, ids
            self.id2int = {
                id: i
                for i, id in enumerate(ids)
            }
            self.ids_head = nn.Linear(self.h_dim, len(self.id2int))

        # optimizer params
        assert lr > 0
        self.lr = lr

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    @abstractmethod
    def forward(self, eegs):
        pass

    def step(self, batch, phase):
        # performs feature extraction
        batch["eegs"] = batch["eegs"].float().to(self.device)
        assert not torch.isnan(batch["eegs"]).any(), f"batched eegs contains nans"
        # filters the data
        # batch["eegs"] = torchaudio.functional.lowpass_biquad(
        #     waveform=batch["eegs"], sample_rate=self.eeg_sampling_rate, cutoff_freq=float(self.max_freq))
        # batch["eegs"] = torchaudio.functional.highpass_biquad(
        #     waveform=batch["eegs"], sample_rate=self.eeg_sampling_rate, cutoff_freq=float(self.min_freq))
        # batch["eegs"] = batch["eegs"].to(torch.float32)
        # generates the spectrogram
        batch["mel_spec"] = self.mel_spectrogrammer(batch["eegs"])  # [b c m t]
        assert not torch.isnan(batch["mel_spec"]).any(
        ), f"batched Mel spectrogram contains nans"
        outs = {
            "metrics": {},
            **self(wf=batch["eegs"], mel_spec=batch["mel_spec"]),
        }

        # classification head
        # assert outs["features"].shape[-1] == self.h_dim, f"{outs['features'].shape} != {self.h_dim}"
        outs["cls_labels"] = batch["labels"].float().to(self.device)
        if "cls_logits" not in outs:
            assert "features" in outs, f"key 'features' not returned by the model. Current keys are {outs.keys()}"
            outs["cls_logits"] = self.cls_head(outs["features"])
        outs["metrics"].update({
            "cls_loss": F.binary_cross_entropy_with_logits(input=outs["cls_logits"], target=outs["cls_labels"]),
            "cls_acc": torchmetrics.functional.accuracy(preds=outs["cls_logits"], target=outs["cls_labels"], task="multilabel" if self.num_labels > 1 else "binary", num_labels=self.num_labels, average="micro"),
            "cls_f1": torchmetrics.functional.f1_score(preds=outs["cls_logits"], target=outs["cls_labels"], task="multilabel" if self.num_labels > 1 else "binary", num_labels=self.num_labels, average="micro"),
        })

        if self.predict_ids:
            # ids head
            assert "subject_id" in batch
            outs["ids_labels"] = torch.as_tensor(
                [self.id2int[id] for id in batch["subject_id"]], dtype=torch.long, device=self.device)
            if "ids_logits" not in outs:
                assert "features" in outs, f"key 'features' not returned by the model. Current keys are {outs.keys()}"
                outs["ids_logits"] = self.ids_head(outs["features"])
            outs["metrics"].update({
                "ids_loss": F.cross_entropy(input=outs["ids_logits"], target=outs["ids_labels"]),
                "ids_acc": torchmetrics.functional.accuracy(preds=outs["ids_logits"], target=outs["ids_labels"], task="multiclass", num_classes=len(self.id2int), average="micro"),
                "ids_f1": torchmetrics.functional.f1_score(preds=outs["ids_logits"], target=outs["ids_labels"], task="multiclass", num_classes=len(self.id2int), average="micro"),
            })

        # computes final loss
        outs["loss"] = sum(
            [v for k, v in outs["metrics"].items() if k.endswith("loss") and v.numel() == 1])

        # logs metrics
        for metric_name, metric_value in outs["metrics"].items():
            # color = Fore.CYAN if phase == "train" else Fore.YELLOW
            self.log(name=f"{metric_name}/{phase}", value=metric_value,
                     prog_bar=any([metric_name.endswith(s)
                                  for s in {"f1"}]),
                     on_step=False, on_epoch=True, batch_size=batch["eegs"].shape[0])
        return outs

    def training_step(self, batch, batch_idx):
        outs = self.step(batch, phase="train")
        return outs

    def validation_step(self, batch, batch_idx):
        outs = self.step(batch, phase="val")
        return outs

    def test_step(self, batch, batch_idx):
        outs = self.step(batch, phase="test")
        return outs
