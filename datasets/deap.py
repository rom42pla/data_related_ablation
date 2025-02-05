import random
from multiprocessing import Pool
from os.path import isdir, join, splitext, basename
from typing import Dict, List, Tuple

import os

import mne
from scipy.signal import butter, filtfilt, welch
import numpy as np
import pandas as pd
import pickle
import einops
import matplotlib.pyplot as plt

from datasets.base_dataset import EEGClassificationDataset


class DEAPDataset(EEGClassificationDataset):
    def __init__(self, path: str, **kwargs):
        super(DEAPDataset, self).__init__(
            name="DEAP",
            path=path,
            sampling_rate=512,
            electrodes=["Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5",
                        "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz", "Fp2", "AF4",
                        "Fz", "F4", "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6",
                        "CP2", "P4", "P8", "PO4", "O2"],
            labels=["valence", "arousal", "dominance", "liking"],
            subject_ids=DEAPDataset.get_subject_ids_static(path=path),
            **kwargs
        )

    def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        global parse_eegs

        self.ratings = pd.read_csv(join(self.path, "participant_ratings.csv"))
        self.ratings['Participant_id'] = self.ratings['Participant_id'].apply(
            lambda x: f"s{int(x):02d}")
        self.ratings.columns = self.ratings.columns.str.lower()

        def parse_eegs(subject_id: str) -> Tuple[List[np.ndarray], List[np.ndarray], str]:
            # preprocessed
            # with open(join(self.path, "data_preprocessed_python", f"{subject_id}.dat"), "rb") as fp:
            #     eegs_raw: Dict[str, np.ndarray] = pickle.load(
            #         fp, encoding='latin1')
            # raw
            eegs_raw = mne.io.read_raw_bdf(
                input_fname=join(self.path, "data_original",
                                 f"{subject_id}.bdf"),
                include=self.electrodes + ["Status"],
                verbose=False
            )
            if (status_channel := "Status") not in eegs_raw.ch_names:
                return
            eegs_events = mne.find_events(
                eegs_raw, stim_channel=status_channel, verbose=False)
            eegs_epochs = mne.Epochs(raw=eegs_raw, events=eegs_events, event_id=4, tmin=0, tmax=60, baseline=(
                0, 0), verbose=False, picks=self.electrodes)
            eegs = eegs_epochs.get_data(verbose=False)
            # filters the data
            # eegs = mne.filter.filter_data(
            #     data=eegs, sfreq=self.sampling_rate, l_freq=self.min_freq, h_freq=self.max_freq, n_jobs=1, verbose=True)
            # eegs = mne.filter.filter_data(
            #     data=eegs, sfreq=self.sampling_rate, l_freq=self.min_freq, h_freq=self.max_freq, n_jobs=1, verbose=False, method="iir", iir_params={"order": 8, "ftype": "butter"})
            
            # freqs, psd_before = self.get_mean_psd(eegs, self.sampling_rate)
            
            eegs = self.bandpass_filter(
                eegs, l_freq=self.min_freq, h_freq=self.max_freq, sampling_rate=self.sampling_rate, order=4)
            
            # freqs, psd_after = welch(
            #     eegs.mean((0, 1)), self.sampling_rate, nperseg=256)
            # freqs, psd_after = self.get_mean_psd(eegs, self.sampling_rate)

            # plt.figure(figsize=(10, 5))
            # plt.semilogy(freqs, psd_before, label='Before Filtering')
            # plt.semilogy(freqs, psd_after, label='After Filtering')
            # plt.xlabel('Frequency (Hz)')
            # plt.ylabel('Power Spectral Density (PSD)')
            # plt.legend()
            # plt.show()
            # raise
            # self.plot_eeg_psd(filtered_data[0], self.sampling_rate)
            eegs = einops.rearrange(
                eegs, "v c t -> v t c")[:, :self.sampling_rate * 60]
            assert list(eegs.shape) == [40, self.sampling_rate * 60, len(
                self.electrodes)], f"{eegs.shape} != {[40, self.sampling_rate * 60, len(self.electrodes)]}"
            # labels
            ratings = self.ratings[self.ratings["participant_id"]
                                   == subject_id].sort_values(by="trial")
            labels = ratings[self.labels].values
            assert list(labels.shape) == [eegs.shape[0], len(self.labels)]
            # converts to lists
            eegs, labels = [x for x in eegs], [x for x in labels]
            # eegs_raw["data"] = einops.rearrange(eegs_raw["data"],
            #                                         "b c s -> b s c")[:, :self.sampling_rate * 60, :32]
            # eegs: List[np.ndarray] = []
            # labels: List[np.ndarray] = []
            # experiments_no = len(eegs_raw["data"])
            # assert experiments_no \
            #     == len(eegs_raw["data"]) == len(eegs_raw["labels"])
            # for i_experiment in range(experiments_no):
            #     # loads the eeg for the experiment
            #     eegs += [eegs_raw["data"][i_experiment]]  # (s c)
            #     # loads the labels for the experiment
            #     labels += [eegs_raw["labels"][i_experiment]]  # (l)
            # eventually discretizes the labels
            labels = [[1 if label > 5 else 0 for label in w] if self.discretize_labels else (w - 1) / 8
                      for w in labels]
            # self.plot_eeg_psd(einops.rearrange(eegs[0], "t c -> c t"), self.sampling_rate)
            return eegs, labels, subject_id

        with Pool(processes=len(self.subject_ids)) as pool:
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

    @staticmethod
    def get_subject_ids_static(path: str) -> List[str]:
        assert isdir(path)
        subject_ids = [basename(splitext(s)[0])
                       for s in os.listdir(join(path, "data_preprocessed_python"))]
        subject_ids.sort()
        return subject_ids


if __name__ == "__main__":
    dataset = DEAPDataset(path=join("..", "..", "..", "datasets", "eeg_emotion_recognition", "deap"),
                          discretize_labels=True, normalize_eegs=True, window_size=1, window_stride=1)
    dataset.plot_sample(i=random.randint(0, len(dataset)))
    dataset.plot_amplitudes_distribution()
    dataset.plot_labels_distribution()
