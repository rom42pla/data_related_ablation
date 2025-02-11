from os.path import join
import torch

from datasets.deap import DEAPDataset
from datasets.high_gamma import HighGammaDataset
from models.edpnet import EDPNet
from models.eegnet import EEGNet
from models.sateer import SATEER


if __name__ == "__main__":
    hg_dataset = HighGammaDataset(path="../../datasets/hg", discretize_labels=True, normalize_eegs=True, window_size=4, window_stride=4, remove_artifacts=True)
    print(f"High Gamma dataset loaded. It contains {len(hg_dataset)} samples.")
    print(hg_dataset[0]["eegs"].shape)
    
    # deap_dataset = DEAPDataset(path="../../datasets/deap", discretize_labels=True, normalize_eegs=True, window_size=2, window_stride=2, remove_artifacts=True)
    # print(f"DEAP loaded. It contains {len(deap_dataset)} samples.")
    # print(deap_dataset[0]["eegs"].shape)
