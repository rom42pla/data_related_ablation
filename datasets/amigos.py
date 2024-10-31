from multiprocessing import Pool
from os.path import isdir, join, splitext, basename
from typing import Dict, List, Tuple

import os

import numpy as np
import scipy.io as sio
import pandas as pd
import pickle
import einops

from datasets.base_class import EEGClassificationDataset


class AMIGOSDataset(EEGClassificationDataset):
    def __init__(self, path: str, **kwargs):
        super(AMIGOSDataset, self).__init__(
            name="AMIGOS",
            path=path,
            sampling_rate=128,
            electrodes=["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
                        "P8", "T8", "FC6", "F4", "F8", "AF4"],
            labels=["arousal", "valence", "dominance", "liking",
                    "familiarity", "neutral", "disgust", "happiness",
                    "surprise", "anger", "fear", "sadness"],
            subject_ids=AMIGOSDataset.get_subject_ids_static(path=path),
            **kwargs
        )

    def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        global parse_eegs

        self.ratings = pd.read_excel(join(self.path, "SelfAsessment.xlsx"))
        self.ratings = self.ratings.iloc[:, [
            0, 1, *[i for i in range(15, 27)]]]
        self.ratings.columns = ["user_id",
                                "video_id", *self.ratings.iloc[0, 2:]]
        self.ratings = self.ratings.iloc[1:]
        self.ratings = self.ratings[["user_id", "video_id", *self.labels]]
        self.ratings["user_id"] = [
            f"P{int(s):02}" for s in self.ratings["user_id"]]
        self.ratings['video_id'] = self.ratings['video_id'].str.replace(
            "'", "").astype(int)
        self.ratings = self.ratings.sort_values(
            by=["user_id", "video_id"]).reset_index(drop=True)

        def parse_eegs(subject_id: str) -> Tuple[List[np.ndarray], List[np.ndarray], str]:
            # parses the eegs
            subject_data = sio.loadmat(join(self.path, "data_original", f"Data_Original_{subject_id}", f"Data_Original_{subject_id}.mat"),
                                       simplify_cells=True)
            eegs_raw = subject_data["EEG_DATA"]
            
            # parses the video ids
            video_ids = [int(s) if str.isnumeric(
                s) else None for s in subject_data["VideoIDs"]]
            
            # checks what samples are not corrupted
            valid_indices = [i for i in range(
                len(eegs_raw)) if video_ids[i] and eegs_raw[i].size > 0]
            eegs_raw = [eegs_raw[i][:, 3:17].astype(
                np.float32) for i in valid_indices]
            video_ids = [int(video_ids[i])
                         for i in valid_indices]
            # filters the data
            eegs = self.bandpass_filter(
                eegs, l_freq=self.min_freq, h_freq=self.max_freq, sampling_rate=self.sampling_rate, order=4)
            
            # parses the labels
            ratings = self.ratings[self.ratings["user_id"]
                                   == subject_id].sort_values(by="video_id").reset_index(drop=True)
            labels = [ratings[ratings["video_id"] == video_id]
                      for video_id in video_ids]
            labels = [row[self.labels].values[0] for row in labels]
            assert len(eegs_raw) == len(labels)
            assert all([v.shape[-1] == len(self.electrodes) for v in eegs_raw])

            # discretizes the labels
            for i_trial, labels_trial in enumerate(labels):
                if self.discretize_labels:
                    labels[i_trial][:5] = np.asarray(
                        [1 if label > 5 else 0 for label in labels_trial[:5]])
                else:
                    labels_trial[labels_trial > 9] = 9
                    labels[i_trial][:5] = (labels_trial[:5] - 1) / 8
                    assert labels[i_trial][:5].min() >= 0
                    assert labels[i_trial][:5].max() <= 9
            labels = [v.astype(np.int32) for v in labels]
            return eegs_raw, labels, subject_id

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
        subject_ids = [basename(splitext(s)[0]).split("_")[-1]
                       for s in os.listdir(join(path, "data_preprocessed"))]
        subject_ids.sort()
        return subject_ids
