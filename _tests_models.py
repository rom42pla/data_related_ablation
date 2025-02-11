import torch

from models.edpnet import EDPNet
from models.eegnet import EEGNet
from models.sateer import SATEER


if __name__ == "__main__":

    eeg_num_channels = 32
    eeg_sampling_rate = 128
    seconds = 1
    eeg_samples = int(eeg_sampling_rate * seconds)
    eeg_sample = torch.rand(1, eeg_num_channels, eeg_samples)
    labels = ["valence", "arousal"]
    print(f"dummy input has shape\t\t{tuple(eeg_sample.shape)}")

    # EEGNet
    eegnet = EEGNet(
        eeg_sampling_rate=eeg_sampling_rate,
        eeg_num_channels=eeg_num_channels,
        eeg_samples=eeg_samples,
        labels=labels,
        predict_ids=False,
    )
    out_eegnet = eegnet(eeg_sample)
    print(f"EEGNet output has shape\t\t{tuple(out_eegnet['cls_logits'].shape)}")

    # EDPNet
    edpnet = EDPNet(
        eeg_sampling_rate=eeg_sampling_rate,
        eeg_num_channels=eeg_num_channels,
        eeg_samples=eeg_samples,
        labels=labels,
        predict_ids=False,
    )
    out_edpnet = edpnet(eeg_sample)
    print(f"EDPNet output has shape\t\t{tuple(out_edpnet['cls_logits'].shape)}")

    # SATEER
    sateer = SATEER(
        eeg_sampling_rate=eeg_sampling_rate,
        eeg_num_channels=eeg_num_channels,
        eeg_samples=eeg_samples,
        labels=labels,
        predict_ids=False,
    )
    out_sateer = sateer(wf=eeg_sample, mel_spec=sateer.mel_spectrogrammer(eeg_sample))
    print(f"SATEER output has shape\t\t{tuple(out_sateer['cls_logits'].shape)}")
