import os
from abc import abstractmethod, ABC
from copy import deepcopy
from math import ceil
from multiprocessing import Pool
from os.path import isdir
from pprint import pprint
from typing import Dict, Optional, Union, List, Tuple

import mne
from scipy import signal
from scipy.signal import butter, filtfilt, welch
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

import numpy as np
import einops
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class EEGClassificationDataset(Dataset, ABC):

    def __init__(
        self,
        path: str,
        name: str,
        sampling_rate: int,
        eeg_electrodes: Union[int, List[str]],
        eog_electrodes: Union[int, List[str]],
        labels: List[str],
        subject_ids: List[str],
        labels_classes: Union[int, List[int]] = 2,
        min_freq: Optional[Union[int, float]] = None,
        max_freq: Optional[Union[int, float]] = None,
        window_size: Optional[Union[float, int]] = 1,
        window_stride: Optional[Union[float, int]] = None,
        drop_last: Optional[bool] = False,
        discretize_labels: bool = False,
        normalize_eegs: bool = True,
        remove_artifacts: bool = True,
    ):
        super().__init__()

        assert isdir(path)
        self.path: str = path
        assert name is None or isinstance(name, str)
        self.name = name

        assert isinstance(sampling_rate, int)
        self.sampling_rate: int = sampling_rate
        self.nyquist_freq = self.sampling_rate // 2 - 1

        assert (
            eeg_electrodes is None
            or isinstance(eeg_electrodes, list)
            or isinstance(eeg_electrodes, int)
        )
        if isinstance(eeg_electrodes, list):
            assert all((isinstance(x, str) for x in eeg_electrodes))
        elif isinstance(eeg_electrodes, int):
            self.electrodes = [f"electrode_{x}" for x in range(eeg_electrodes)]
        self.electrodes: List[str] = eeg_electrodes
        self.eog_electrodes = eog_electrodes
        self.num_channels = len(self.electrodes)

        if min_freq is None:
            min_freq = 1
        if max_freq is None:
            max_freq = self.nyquist_freq
        self.min_freq = max(1, min_freq)
        self.max_freq = min(self.nyquist_freq, max_freq)
        assert 0 <= self.min_freq < self.max_freq, f"got 0 <= {self.min_freq} < {self.max_freq}"

        assert isinstance(labels, list)
        assert all((isinstance(x, str) for x in labels))
        self.labels: List[str] = labels
        assert isinstance(labels_classes, int) or isinstance(labels_classes, list), \
            f"the labels classes must be a list of integers or a positive integer, not {labels_classes}"
        if isinstance(labels_classes, list):
            assert all([isinstance(labels_class, int) for labels_class in labels_classes]), \
                f"if the name of the labels are given ({labels_classes}), they must all be strings"
            assert len(labels_classes) == len(labels)
            self.labels_classes = labels_classes
        else:
            assert labels_classes > 0, \
                f"there must be a positive number of classes, not {labels_classes}"
            self.labels_classes = [labels_classes for _ in self.labels]

        assert isinstance(subject_ids, list)
        assert all((isinstance(x, str) for x in subject_ids))
        self.subject_ids: List[str] = subject_ids
        self.subject_ids.sort()

        if self.name == {"High Gamma"}:
            assert window_size == 4, "window_size must be 4"
            assert window_stride == 4, "window_stride must be 4"
        assert window_size > 0
        self.window_size: float = float(window_size)  # s
        self.samples_per_window: int = int(np.floor(self.sampling_rate * self.window_size))
        assert window_stride is None or window_stride > 0
        if window_stride is None:
            window_stride = deepcopy(self.window_size)
        self.window_stride: float = float(window_stride)  # s
        self.samples_per_stride: int = int(np.floor(self.sampling_rate * self.window_stride))
        assert isinstance(drop_last, bool)
        self.drop_last: bool = drop_last

        assert isinstance(discretize_labels, bool)
        self.discretize_labels: bool = discretize_labels
        assert isinstance(remove_artifacts, bool)
        self.remove_artifacts: bool = remove_artifacts
        assert isinstance(normalize_eegs, bool)
        self.normalize_eegs: bool = normalize_eegs

        self.eegs_data, self.labels_data, self.subject_ids_data = self.load_data()
        # self.eegs_data is a list of np.ndarray of shape (t c)
        assert len(self.eegs_data) == len(self.labels_data) == len(self.subject_ids_data)
        
        # updates subject_ids
        self.subject_ids= sorted(set(self.subject_ids_data))

        if self.name != "High Gamma":
            assert all([eeg.shape[1] == len(self.electrodes + self.eog_electrodes) for eeg in self.eegs_data])
            if self.remove_artifacts:
                self.artifact_removal_inplace(self.eegs_data)
            if self.normalize_eegs:
                self.eegs_data = self.normalize(self.eegs_data)
            # removes eog channels
            for i_eeg in range(len(self.eegs_data)):
                self.eegs_data[i_eeg] = self.eegs_data[i_eeg][:, :len(self.electrodes)]
            # eventually filters data
            self.bands_filter_inplace(self.eegs_data)
        assert all([eeg.shape[1] == len(self.electrodes) for eeg in self.eegs_data]), self.eegs_data[0].shape
        self.windows = self.get_windows()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> Dict[str, Union[int, str, np.ndarray]]:
        window = self.windows[i]
        eegs = self.eegs_data[window["experiment"]][window["start"]:window["end"]]
        # eventually pad the eegs
        if eegs.shape[0] != self.samples_per_window:
            eegs = np.concatenate([eegs,
                                   np.zeros([self.samples_per_window - eegs.shape[0], eegs.shape[1]])],
                                  axis=0)
        # check format
        eegs = einops.rearrange(eegs, "t c -> c t")
        assert eegs.shape == (len(self.electrodes), self.samples_per_window), f"expected [{len(self.electrodes)}, {self.samples_per_window}], got {eegs.shape}" 
        # asserts that there are no frequencies in the selected ranges
        # self.plot_eeg_psd(eegs, self.sampling_rate)
        # psd = self.get_psd(eegs, sampling_rate=self.sampling_rate)
        return {
            "sampling_rates": self.sampling_rate,
            "subject_id": window["subject_id"],
            "eegs": torch.from_numpy(eegs.astype(np.float32)),
            "labels": torch.from_numpy(window["labels"])
        }

    def prepare_data(self) -> None:
        pass

    @staticmethod
    @abstractmethod
    def get_subject_ids_static(path: str):
        pass

    @abstractmethod
    def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        pass

    def artifact_removal_inplace(self, eegs, notch_freq=50, ica_n_components=0.96):
        is_list = isinstance(eegs, list)
        # apply ICA for eog artifact removal
        for i in tqdm(range(len(eegs)), desc="Removing artifacts", disable=not is_list):
            if is_list:
                # create MNE raw object
                info = mne.create_info(ch_names=self.electrodes, sfreq=self.sampling_rate, ch_types="eeg", verbose=False)
                raw = mne.io.RawArray(eegs[i].T, info, verbose=False)
            else:
                raw = eegs
            # apply Notch filter at 50 Hz
            raw = raw.notch_filter(freqs=notch_freq, fir_design="firwin", n_jobs=1, verbose=False)

            # apply ICA for eog artifact removal
            raw = raw.filter(l_freq=1., h_freq=None, verbose=False)
            ica = mne.preprocessing.ICA(n_components=ica_n_components, random_state=42, max_iter=1000, verbose=False)
            ica.fit(raw, verbose=False)
            eog_indices, _ = ica.find_bads_eog(raw, ch_name=self.eog_electrodes[1], verbose=False)  # detect eye-related artifacts
            ica.exclude = eog_indices # mark artifacts for removal
            raw = ica.apply(raw, verbose=False)  # apply ICA in place

            if is_list:
                # write the cleaned data back into the original NumPy array
                eegs[i] = raw.get_data().T
            # write the cleaned data back into the original NumPy array
            else:
                return raw

    def normalize(self, eegs: List[np.ndarray]):
        # scales to zero mean and unit variance
        for i_experiment, experiment in enumerate(eegs):
            # scales to zero mean and unit variance
            # experiment_scaled = (experiment - experiment.mean(axis=0)) / experiment.std(axis=0)
            scaler = mne.decoding.Scaler(info=mne.create_info(ch_names=self.electrodes, sfreq=self.sampling_rate,
                                                              verbose=False, ch_types="eeg"),
                                         scalings="mean")
            experiment_scaled = einops.rearrange(
                scaler.fit_transform(einops.rearrange(experiment, "s c -> () c s")),
                "b c s -> s (b c)"
            )
            experiment_scaled = np.nan_to_num(experiment_scaled)
            # normalizes between -1 and 1
            experiment_scaled = 2 * ((experiment_scaled - experiment_scaled.min(axis=0)) /
                                     (experiment_scaled.max(axis=0) - experiment_scaled.min(axis=0))) - 1
            eegs[i_experiment] = experiment_scaled
        return eegs

    def bands_filter_inplace(self, eegs):
        is_list = isinstance(eegs, list)
        for i in tqdm(range(len(eegs)), desc=f"Filtering data between {self.min_freq} and {self.max_freq}Hz", disable=not is_list):
            if is_list:
                # create MNE raw object
                info = mne.create_info(ch_names=self.electrodes, sfreq=self.sampling_rate, ch_types="eeg", verbose=False)
                raw = mne.io.RawArray(eegs[i].T, info, verbose=False)
            else:
                raw = eegs

            # apply ICA for eog artifact removal
            raw = raw.filter(l_freq=self.min_freq, h_freq=self.max_freq, verbose=False)

            if is_list:
                # write the cleaned data back into the original NumPy array
                eegs[i] = raw.get_data().T
            # write the cleaned data back into the original NumPy array
            else:
                return raw

    @staticmethod
    def has_nans(array: np.ndarray) -> bool:
        return np.isnan(array).any()

    def get_windows(self) -> List[Dict[str, Union[int, str]]]:
        windows: List[Dict[str, Union[int, str]]] = []
        for i_experiment in range(len(self.eegs_data)):
            for i_window_start in range(0,
                                        len(self.eegs_data[i_experiment]),
                                        self.samples_per_stride):
                window = {
                    "experiment": i_experiment,
                    "start": i_window_start,
                    "end": i_window_start + self.samples_per_window,
                    "subject_id": self.subject_ids_data[i_experiment],
                    "labels": np.asarray(self.labels_data[i_experiment]),
                }
                if self.drop_last is True and (window["end"] - window["start"]) != self.samples_per_window:
                    continue
                windows += [window]
        return windows

    @staticmethod
    def bandpass_filter(eegs, l_freq, h_freq, sampling_rate, order=8):
        nyquist = 0.5 * sampling_rate
        low = l_freq / nyquist
        high = h_freq / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, eegs, axis=-1)

    @staticmethod
    def get_mean_psd(eegs, sampling_rate):
        assert len(
            eegs.shape) == 3, f"data must be (epochs, channels, times), got {eegs.shape}"
        freqs, psd = welch(
            eegs.mean((0, 1)), sampling_rate, nperseg=sampling_rate//2)
        return freqs, psd

    def plot_sample(
            self,
            i: int,
            title: str = f"sample",
            scale: Union[int, float] = 5,
    ) -> None:
        assert isinstance(i, int) and i >= 0
        cols = 4
        rows = len(self.electrodes) // cols
        fig, axs = plt.subplots(nrows=rows, ncols=cols,
                                figsize=(scale * cols // 2, scale * rows // 4),
                                tight_layout=True)
        # fig.suptitle(title)
        for i_ax, ax in enumerate(axs.flat):
            if i_ax >= len(self.electrodes):
                ax.set_visible(False)
                continue
            axs.flat[i_ax].plot(self[i]["eegs"][:, i_ax])
            axs.flat[i_ax].set_title(self.electrodes[i_ax])
            axs.flat[i_ax].set_ylim(self[i]["eegs"].min(), self[i]["eegs"].max())
        plt.show()
        fig.clf()
        # raw_mne_array = mne.io.RawArray(einops.rearrange(self[i]["eegs"], "s c -> c s"),
        #                                 info=mne.create_info(ch_names=self.electrodes, sfreq=self.sampling_rate,
        #                                                      verbose=False, ch_types="eeg"), verbose=False)
        # raw_mne_array.plot()

    def plot_subjects_distribution(self) -> None:
        subject_indices_samples = sorted([s["subject_id"]
                                          for s in self])
        subject_ids_samples = [self.subject_ids[i]
                               for i in subject_indices_samples]
        fig, ax = plt.subplots(1, 1,
                               figsize=(15, 5),
                               tight_layout=True)
        sns.countplot(x=subject_ids_samples,
                      palette="rocket", ax=ax)
        plt.show()
        fig.clf()

    def plot_labels_distribution(
            self,
            title: str = "distribution of labels",
            scale: Union[int, float] = 4,
    ):
        print("retrieving labels")
        labels = np.concatenate([window["labels"] for window in self.windows])
        unique_values, counts = np.unique(labels, return_counts=True)
        counts = counts / counts.sum()

        # Create a new figure
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(scale, scale))

        # Plot the distribution of the labels
        ax.bar(unique_values, counts)
        ax.set_xticks(unique_values)
        ax.set_xlabel('Label')
        ax.set_ylabel('Frequency')
        fig.suptitle(title)

        # if self.discretize_labels:
        #     cols = min(8, len(self.labels))
        #     rows = 1 if (len(self.labels) <= 8) else ceil(len(self.labels) / 8)
        #     fig, axs = plt.subplots(nrows=rows, ncols=cols,
        #                             figsize=(scale * cols, scale * rows),
        #                             tight_layout=True)
        #     fig.suptitle(title)
        #     labels_data = np.stack([x["labels"] for x in self])
        #     for i_ax, ax in enumerate(axs.flat):
        #         if i_ax >= len(self.labels):
        #             ax.set_visible(False)
        #             continue
        #         unique_labels = np.unique(labels_data[:, i_ax])
        #         sizes = [np.count_nonzero(labels_data[:, i_ax] == unique_label)
        #                  for unique_label in unique_labels]
        #         axs.flat[i_ax].pie(sizes, labels=unique_labels, autopct='%1.1f%%',
        #                            shadow=False)
        #         axs.flat[i_ax].axis('equal')
        #         axs.flat[i_ax].set_title(self.labels[i_ax])
        # else:
        #     # builds the dataframe
        #     df = pd.DataFrame()
        #     for x in self:
        #         for i_label, label in enumerate(self.labels):
        #             df = pd.concat([df, pd.DataFrame([{
        #                 "label": label,
        #                 "value": x["labels"][i_label]
        #             }])], ignore_index=True).sort_values(by="value", ascending=True)
        #     fig, axs = plt.subplots(nrows=1, ncols=len(self.labels),
        #                             figsize=(scale * len(self.labels), scale),
        #                             tight_layout=True)
        #     fig.suptitle(title)
        #     # plots
        #     for i_label, label in enumerate(self.labels):
        #         ax = axs[i_label]
        #         sns.histplot(
        #             data=df[df["label"] == label],
        #             bins=16,
        #             palette="rocket",
        #             ax=ax
        #         )
        #         ax.set_xlabel(label)
        #         ax.set_ylabel("count")
        #         ax.get_legend().remove()
        #     # adjusts the ylim
        #     max_ylim = max([ax.get_ylim()[-1] for ax in axs])
        #     for ax in axs:
        #         ax.set_ylim([0, max_ylim])
        plt.show()
        fig.clf()

    def plot_amplitudes_distribution(
            self,
            title: str = "distribution of amplitudes",
            scale: Union[int, float] = 4,
    ):
        cols: int = 8
        rows: int = ceil(len(self.electrodes) / cols)
        fig, axs = plt.subplots(nrows=rows, ncols=cols,
                                figsize=(scale * cols, scale * rows),
                                tight_layout=True)
        if self.name is not None:
            title = f"{title} - {self.name} dataset"
        fig.suptitle(title)
        for i_electrode, ax in enumerate(axs.flat):
            if i_electrode >= len(self.electrodes):
                ax.set_visible(False)
                continue
            ax.hist(
                np.concatenate([x["eegs"][:, i_electrode] for x in self]),
                bins=32
            )
            ax.set_title(self.electrodes[i_electrode])
            ax.set_xlabel("mV")
            ax.set_ylabel("count")
            ax.set_yscale("log")
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        # adjusts the ylim
        max_ylim = max([ax.get_ylim()[-1] for ax in axs.flat])
        for ax in axs.flat:
            ax.set_ylim([None, max_ylim])
        plt.show()
        fig.clf()

    def plot_eeg_psd(self, title="PSD"):
        info = mne.create_info(ch_names=self.electrodes,
                            sfreq=self.sampling_rate,
                            ch_types='eeg', verbose=False)
        raw = mne.io.RawArray(np.concatenate([eeg for eeg in self.eegs_data], axis=-1), info, verbose=False)

        fig, ax = plt.subplots(1, 1)
        raw.compute_psd().plot(xscale="log", axes=ax)
        fig.suptitle(title)
        plt.show()
