import torch
from torch import nn
import lightning as pl
from transformers import AutoImageProcessor, Dinov2Model, Dinov2Config
from torch.utils.data import DataLoader, Dataset
from typing import Union, List

# Placeholder dataset class for EEG data


# class EEGDataset(Dataset):
#     def __init__(self, data, labels, image_processor):
#         self.data = data
#         self.labels = labels
#         self.image_processor = image_processor

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         image = self.data[idx]
#         label = self.labels[idx]
#         inputs = self.image_processor(image, return_tensors="pt")
#         return inputs['pixel_values'].squeeze(0), torch.tensor(label)

# Lightning module for EEG classification using DINOv2


class EEGClassificationModel(pl.LightningModule):
    def __init__(self,
                 eeg_sampling_rate: int,
                 window_size: Union[int, float],
                 window_stride: Union[int, float],
                 mels: int = 8,
                 min_freq: int = 0,
                 max_freq: int = 50,
                 
                 num_labels: int,
                 num_channels: int,
                 patch_size: int = 1,
                 hidden_size: int = 256,
                 num_hidden_layers: int = 2,
                 num_attention_heads: int = 8,
                 hidden_act: str = "relu",
                 dropout: float = 0.1,
                 lr: float = 1e-3):
        super(EEGClassificationModel, self).__init__()
        self.save_hyperparameters()

        # model params
        self.num_labels = num_labels
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.dropout = dropout

        # self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Dinov2Model(Dinov2Config(
            num_channels=self.num_channels,
            patch_size=self.patch_size,
            hidden_size=self.hidden_size,
            hidden_act=self.hidden_act,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            hidden_dropout_prob=self.dropout,
            attention_probs_dropout_prob=self.dropout,
        ))
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, pixel_values):
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
        last_hidden_states = outputs.last_hidden_state
        # assuming classification token is at position 0
        logits = self.classifier(last_hidden_states[:, 0, :])
        return logits

    def training_step(self, batch, batch_idx):
        pixel_values, labels = batch
        logits = self.forward(pixel_values)
        loss = self.loss_fn(logits, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values, labels = batch
        logits = self.forward(pixel_values)
        loss = self.loss_fn(logits, labels)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        pixel_values, labels = batch
        logits = self.forward(pixel_values)
        loss = self.loss_fn(logits, labels)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer


# Example usage
if __name__ == '__main__':
    model_name = "facebook/dinov2-base"
    num_classes = 4  # Define the number of classes for your classification task

    # train_dataset = EEGDataset(
    #     train_data, train_labels, AutoImageProcessor.from_pretrained(model_name))
    # val_dataset = EEGDataset(val_data, val_labels,
    #                          AutoImageProcessor.from_pretrained(model_name))
    # test_dataset = EEGDataset(test_data, test_labels,
    #                           AutoImageProcessor.from_pretrained(model_name))

    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32)
    # test_loader = DataLoader(test_dataset, batch_size=32)

    model = EEGClassificationModel(
        num_labels=num_classes, num_channels=32,
    )
    x = torch.randn(8, 32, 256)
    print(model)
    print(model(x))
    # trainer = pl.Trainer(max_epochs=10)
    # trainer.fit(model, train_loader, val_loader)
    # trainer.test(model, test_loader)
