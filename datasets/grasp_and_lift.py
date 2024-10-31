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


class GraspAndLiftDataset(EEGClassificationDataset):
    def __init__(self, path: str, **kwargs):
        super(GraspAndLiftDataset, self).__init__(
            name="Grasp and Lift",
            path=path,
            sampling_rate=500,
            electrodes=['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2',
                        'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6',
                        'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10'],
            # labels=["HandStart", "FirstDigitTouch", "BothStartLoadPhase",
            #         "LiftOff", "Replace", "BothReleased"],
            labels=["hand_grasp"],
            labels_classes=2,
            subject_ids=GraspAndLiftDataset.get_subject_ids_static(path=path),
            **kwargs
        )

    def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        self.samples_per_window = floor(2 * self.sampling_rate * 0.15)
        self.samples_per_stride = self.samples_per_window // 2
        global parse_eegs

        def parse_eegs(subject_id: str) -> Tuple[List[np.ndarray], List[np.ndarray], str]:
            # parses data and events
            subject_data = pd.concat([pd.read_csv(join(
                self.path, "train", f"{subject_id}_series{i}_data.csv")) for i in range(1, 9)]).set_index("id", drop=True)
            events = pd.concat([pd.read_csv(join(
                self.path, "train", f"{subject_id}_series{i}_events.csv")) for i in range(1, 9)]).set_index("id", drop=True)
            # transforms the raw data into mne structures
            eegs_raw = mne.io.RawArray(
                data=einops.rearrange(subject_data.values, "t c -> c t"),
                info=mne.create_info(
                    ch_names=self.electrodes,
                    sfreq=self.sampling_rate,
                    ch_types="eeg",
                    verbose=False,
                ),
                verbose=False,
            )
            events_times = np.arange(
                start=self.samples_per_stride,
                stop=len(subject_data)-self.samples_per_stride,
                step=self.samples_per_stride)
            epochs = mne.Epochs(
                raw=eegs_raw,
                events=np.stack([
                    events_times,
                    np.zeros_like(events_times),
                    np.zeros_like(events_times),
                ], axis=-1).astype(np.int32),
                tmin=-0.15,
                tmax=0.15,
                baseline=None,
                preload=True,
                verbose=False,
            )
            eegs = epochs.get_data(copy=True)
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
            # builds the structures
            eegs_raw = [e[:, :self.samples_per_window] for e in eegs.astype(
                np.float32)]
            labels = [events.values[i].astype(np.int32) for i in events_times]
            # from multilabel to binary
            labels = [np.asarray([1 if np.any(label[1:5]) else 0]) for label in labels]
            # print(np.unique(np.concatenate(labels).flatten(), return_counts=True))
            return eegs_raw, labels, subject_id

        with Pool(processes=len(self.subject_ids)) as pool:
            data_pool = pool.map(
                parse_eegs, [s_id for s_id in self.subject_ids])
            data_pool = [d for d in data_pool if d is not None]
            eegs: List[int] = [e for eeg_lists, _,
                               _ in data_pool for e in eeg_lists]
            labels: List[np.ndarray] = [l for _, labels_lists, _ in data_pool
                                        for l in labels_lists]
            subject_ids: List[str] = [s_id for eegs_lists, _, subject_id, in data_pool
                                      for s_id in [subject_id] * len(eegs_lists)]
            # self.eegs_raw = {
            #     subject_id: eegs_raw for _, _, subject_id, eegs_raw, _ in data_pool
            # }
            # self.epochs = {
            #     subject_id: epochs for _, _, subject_id, _, epochs in data_pool
            # }
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

        subject_ids = [basename(splitext(s)[0])
                       for s in os.listdir(join(path, "train"))]
        subject_ids = list(
            {s.split("_")[0] for s in subject_ids if re.fullmatch("subj[0-9]+.*", s)})
        subject_ids.sort()
        return subject_ids
