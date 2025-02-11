from multiprocessing import Pool
from os.path import isdir, join, splitext, basename
from typing import Dict, List, Tuple, Union
import re
import mne
import os
from math import floor

import numpy as np
import scipy.io as sio
import pandas as pd
import pickle
import einops
import torch

from datasets.base_dataset import EEGClassificationDataset


class HighGammaDataset(EEGClassificationDataset):
    def __init__(self, path: str, split="train", **kwargs):
        assert split in ["train", "test"], f"got {split}"
        self.split = split
        
        super(HighGammaDataset, self).__init__(
            name="High Gamma",
            path=path,
            sampling_rate=500,
            eeg_electrodes=['EEG Fp1', 'EEG Fp2', 'EEG Fpz', 'EEG F7', 'EEG F3', 'EEG Fz', 'EEG F4', 'EEG F8', 'EEG FC5', 'EEG FC1', 'EEG FC2', 'EEG FC6', 'EEG M1', 'EEG T7', 'EEG C3', 'EEG Cz', 'EEG C4', 'EEG T8', 'EEG M2', 'EEG CP5', 'EEG CP1', 'EEG CP2', 'EEG CP6', 'EEG P7', 'EEG P3', 'EEG Pz', 'EEG P4', 'EEG P8', 'EEG POz', 'EEG O1', 'EEG Oz', 'EEG O2', 'EEG AF7', 'EEG AF3', 'EEG AF4', 'EEG AF8', 'EEG F5', 'EEG F1', 'EEG F2', 'EEG F6', 'EEG FC3', 'EEG FCz', 'EEG FC4', 'EEG C5', 'EEG C1', 'EEG C2', 'EEG C6', 'EEG CP3', 'EEG CPz', 'EEG CP4', 'EEG P5', 'EEG P1', 'EEG P2', 'EEG P6', 'EEG PO5', 'EEG PO3', 'EEG PO4', 'EEG PO6', 'EEG FT7', 'EEG FT8', 'EEG TP7', 'EEG TP8', 'EEG PO7', 'EEG PO8', 'EEG FT9', 'EEG FT10', 'EEG TPP9h', 'EEG TPP10h', 'EEG PO9', 'EEG PO10', 'EEG P9', 'EEG P10', 'EEG AFF1', 'EEG AFz', 'EEG AFF2', 'EEG FFC5h', 'EEG FFC3h', 'EEG FFC4h', 'EEG FFC6h', 'EEG FCC5h', 'EEG FCC3h', 'EEG FCC4h', 'EEG FCC6h', 'EEG CCP5h', 'EEG CCP3h', 'EEG CCP4h', 'EEG CCP6h', 'EEG CPP5h', 'EEG CPP3h', 'EEG CPP4h', 'EEG CPP6h', 'EEG PPO1', 'EEG PPO2', 'EEG I1', 'EEG Iz', 'EEG I2', 'EEG AFp3h', 'EEG AFp4h', 'EEG AFF5h', 'EEG AFF6h', 'EEG FFT7h', 'EEG FFC1h', 'EEG FFC2h', 'EEG FFT8h', 'EEG FTT9h', 'EEG FTT7h', 'EEG FCC1h', 'EEG FCC2h', 'EEG FTT8h', 'EEG FTT10h', 'EEG TTP7h', 'EEG CCP1h', 'EEG CCP2h', 'EEG TTP8h', 'EEG TPP7h', 'EEG CPP1h', 'EEG CPP2h', 'EEG TPP8h', 'EEG PPO9h', 'EEG PPO5h', 'EEG PPO6h', 'EEG PPO10h', 'EEG POO9h', 'EEG POO3h', 'EEG POO4h', 'EEG POO10h', 'EEG OI1h', 'EEG OI2h'],
            eog_electrodes=['EOG EOGh', 'EOG EOGv'],
            labels=["action"],
            labels_classes=4,
            subject_ids=HighGammaDataset.get_subject_ids_static(path=path),
            **kwargs
        )

    def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        global parse_eegs
        
        def parse_eegs(subject_id: str) -> Tuple[List[np.ndarray], List[np.ndarray], str]:
            eegs_raw = mne.io.read_raw_edf(
                input_fname=join(self.path, self.split, f"{subject_id}.edf"),
                include=self.electrodes + self.eog_electrodes,
                preload=True,
                verbose=False,
            )
            if self.remove_artifacts:
                eegs_raw = self.artifact_removal_inplace(eegs_raw)
            # eventually filters data
            eegs_raw = self.bands_filter_inplace(eegs_raw)
            eegs_events, event_dict = mne.events_from_annotations(eegs_raw, event_id='auto', verbose=False)
            eegs_epochs = mne.Epochs(raw=eegs_raw, events=eegs_events, event_id=event_dict, baseline=(0,0), verbose=False, tmin=0, tmax=4, picks=self.electrodes)
            
            eegs = eegs_epochs.get_data(verbose=False)
            if self.normalize_eegs:
                eegs = einops.rearrange(eegs, "b c s-> b s c")
                eegs = self.normalize(eegs)
                eegs = einops.rearrange(eegs, "b s c -> b c s")
            labels = eegs_events[:, 2] - 1
            eegs = einops.rearrange(
                eegs, "v c t -> v t c")[:, :self.sampling_rate * 4]
            assert list(eegs.shape)[1:] == [self.sampling_rate * 4, len(
                self.electrodes)], f"{eegs.shape[1:]} != {[self.sampling_rate * 4, len(self.electrodes)]}"
            # labels
            assert list(labels.shape) == [eegs.shape[0]]
            # converts to lists
            eegs, labels = [x for x in eegs], [x for x in labels]
            return eegs, labels, subject_id

        with Pool(processes=2) as pool:
            data_pool = pool.map(
                parse_eegs, [s_id for s_id in self.subject_ids])
            data_pool = [d for d in data_pool if d is not None]
            eegs: List[np.ndarray] = [e for eeg_lists, _, _ in data_pool
                                      for e in eeg_lists]
            labels: List[np.ndarray] = [l for _, labels_lists, _ in data_pool
                                        for l in labels_lists]
            subject_ids: List[str] = [s_id for eegs_lists, _, subject_id in data_pool
                                      for s_id in [subject_id] * len(eegs_lists)]
        assert len(eegs) == len(labels) == len(subject_ids)
        return eegs, labels, subject_ids

    def get_windows(self) -> List[Dict[str, Union[int, str]]]:
        windows: List[Dict[str, Union[int, str]]] = []
        for i_window in range(len(self.eegs_data)):
            window = {
                "experiment": 0,
                "start": i_window,
                "end": i_window,
                "subject_id": self.subject_ids_data[i_window],
                "labels": np.asarray(self.labels_data[i_window]),
            }
            windows += [window]
        return windows

    def __getitem__(self, i: int) -> Dict[str, Union[int, str, np.ndarray]]:
        window = self.windows[i]
        eegs = self.eegs_data[window["start"]]
        return {
            "sampling_rates": self.sampling_rate,
            "subject_id": window["subject_id"],
            "eegs": torch.from_numpy(eegs),
            "labels": torch.from_numpy(window["labels"])
        }

    @staticmethod
    def get_subject_ids_static(path: str) -> List[str]:
        assert isdir(path)

        subject_ids = list({basename(splitext(s)[0])
                            for s in os.listdir(join(path, "train"))
                            if s.endswith(".edf")})
        subject_ids.sort()
        return subject_ids
