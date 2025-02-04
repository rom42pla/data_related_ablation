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

from datasets.base_class import EEGClassificationDataset


class HighGammaDataset(EEGClassificationDataset):
    def __init__(self, path: str, **kwargs):
        super(HighGammaDataset, self).__init__(
            name="High Gamma",
            path=path,
            sampling_rate=500,
            electrodes=['EEG Fp1', 'EEG Fp2', 'EEG Fpz', 'EEG F7', 'EEG F3', 'EEG Fz', 'EEG F4', 'EEG F8', 
                        'EEG FC5', 'EEG FC1', 'EEG FC2', 'EEG FC6', 'EEG M1', 'EEG T7', 'EEG C3', 'EEG Cz', 
                        'EEG C4', 'EEG T8', 'EEG M2', 'EEG CP5', 'EEG CP1', 'EEG CP2', 'EEG CP6', 'EEG P7', 
                        'EEG P3', 'EEG Pz', 'EEG P4', 'EEG P8', 'EEG POz', 'EEG O1', 'EEG Oz', 'EEG O2', 
                        'EOG EOGh', 'EOG EOGv', 'EMG EMG_RH', 'EMG EMG_LH', 'EMG EMG_RF', 'EEG AF7', 'EEG AF3', 'EEG AF4', 
                        'EEG AF8', 'EEG F5', 'EEG F1', 'EEG F2', 'EEG F6', 'EEG FC3', 'EEG FCz', 'EEG FC4', 'EEG C5', 
                        'EEG C1', 'EEG C2', 'EEG C6', 'EEG CP3', 'EEG CPz', 'EEG CP4', 'EEG P5', 'EEG P1', 'EEG P2', 
                        'EEG P6', 'EEG PO5', 'EEG PO3', 'EEG PO4', 'EEG PO6', 'EEG FT7', 'EEG FT8', 'EEG TP7', 'EEG TP8', 'EEG PO7', 'EEG PO8', 'EEG FT9', 'EEG FT10', 'EEG TPP9h',
                        'EEG TPP10h', 'EEG PO9', 'EEG PO10', 'EEG P9', 'EEG P10', 'EEG AFF1', 'EEG AFz', 'EEG AFF2', 
                        'EEG FFC5h', 'EEG FFC3h', 'EEG FFC4h', 'EEG FFC6h', 'EEG FCC5h', 'EEG FCC3h', 'EEG FCC4h', 
                        'EEG FCC6h', 'EEG CCP5h', 'EEG CCP3h', 'EEG CCP4h', 'EEG CCP6h', 'EEG CPP5h', 'EEG CPP3h', 
                        'EEG CPP4h', 'EEG CPP6h', 'EEG PPO1', 'EEG PPO2', 'EEG I1', 'EEG Iz', 'EEG I2', 'EEG AFp3h', 
                        'EEG AFp4h', 'EEG AFF5h', 'EEG AFF6h', 'EEG FFT7h', 'EEG FFC1h', 'EEG FFC2h', 'EEG FFT8h', 
                        'EEG FTT9h', 'EEG FTT7h', 'EEG FCC1h', 'EEG FCC2h', 'EEG FTT8h', 'EEG FTT10h', 'EEG TTP7h', 
                        'EEG CCP1h', 'EEG CCP2h', 'EEG TTP8h', 'EEG TPP7h', 'EEG CPP1h', 'EEG CPP2h', 'EEG TPP8h', 
                        'EEG PPO9h', 'EEG PPO5h', 'EEG PPO6h', 'EEG PPO10h', 'EEG POO9h', 'EEG POO3h', 'EEG POO4h', 
                        'EEG POO10h', 'EEG OI1h', 'EEG OI2h'],
            labels=["hand_grasp"],
            labels_classes=2,
            subject_ids=HighGammaDataset.get_subject_ids_static(path=path),
            **kwargs
        )

    def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        self.samples_per_window = floor(2 * self.sampling_rate * 0.15)
        self.samples_per_stride = self.samples_per_window // 2
        global parse_eegs

        def parse_eegs(subject_id: str) -> Tuple[List[np.ndarray], List[np.ndarray], str]:
            subject_data = mne.io.read_raw_edf(
                input_fname=join(self.path, "train", f"{subject_id}.edf"), 
                include=self.electrodes,
                verbose=False
                )
            events, events_labels = mne.events_from_annotations(subject_data, verbose=False)
            epochs = mne.Epochs(
                raw=subject_data,
                events=events,
                tmin=0,
                tmax=4,
                baseline=None,
                preload=True,
                verbose=False,
            )
            labels = [v for v in epochs.events[:, -1]]
            eegs = epochs.get_data(copy=True, verbose=False)[:, :, :self.sampling_rate * 4]
            assert len(eegs) == len(labels), f"{len(eegs)} != {len(labels)}"
            # perform operations on data
            if self.normalize_eegs:
                scaler = mne.decoding.Scaler(info=mne.create_info(ch_names=self.electrodes, sfreq=self.sampling_rate,
                                                                  verbose=False, ch_types="eeg"),
                                             scalings="mean")
                eegs = scaler.fit_transform(eegs)
                eegs = np.nan_to_num(eegs)
                # normalizes between -1 and 1
                eegs = 2 * ((eegs - eegs.min(axis=(0, 2), keepdims=True)) /
                            (eegs.max(axis=(0, 2), keepdims=True) - eegs.min(axis=(0, 2), keepdims=True))) - 1
            # filters the data
            eegs = self.bandpass_filter(
                eegs, l_freq=self.min_freq, h_freq=self.max_freq, sampling_rate=self.sampling_rate, order=4)
            # from multilabel to binary
            labels = [np.asarray([1 if label in {1, 2, 4} else 0])
                      for label in labels]
            return eegs, labels, subject_id

        with Pool(processes=4) as pool:
            data_pool = pool.map(
                parse_eegs, [s_id for s_id in self.subject_ids])
            data_pool = [d for d in data_pool if d is not None]
            eegs: List[int] = [e for eeg_lists, _,
                               _ in data_pool for e in eeg_lists]
            labels: List[np.ndarray] = [l for _, labels_lists, _ in data_pool
                                        for l in labels_lists]
            subject_ids: List[str] = [s_id for eegs_lists, _, subject_id, in data_pool
                                      for s_id in [subject_id] * len(eegs_lists)]
        assert len(eegs) == len(labels) == len(subject_ids)
        return eegs, labels, subject_ids

    def normalize(self, eegs: List[np.ndarray]):
        return eegs

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
