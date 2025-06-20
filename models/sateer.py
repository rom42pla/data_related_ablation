from typing import Optional, OrderedDict, Union, List, Dict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import torchaudio
import einops
from einops.layers.torch import Rearrange
import random
import math
from models.base_model import EEGClassificationModel


class GetSinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, max_position_embeddings: int = 1024):
        super().__init__()
        assert isinstance(max_position_embeddings, int) and max_position_embeddings >= 1
        self.max_position_embeddings = max_position_embeddings

    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 3  # (b s c)
        sequence_length, embeddings_dim = self.max_position_embeddings, x.shape[-1]
        pe = torch.zeros(sequence_length, embeddings_dim, device=x.device)
        position = torch.arange(0, sequence_length, device=x.device).unsqueeze(1)
        div_term = torch.exp(
            (
                torch.arange(0, embeddings_dim, 2, dtype=torch.float, device=x.device)
                * -(math.log(10000.0) / embeddings_dim)
            )
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        del position, div_term
        pe = pe[: x.shape[1]].repeat(x.shape[0], 1, 1)[:, : x.shape[1]]
        assert pe.shape == x.shape
        return pe


class GetLearnedPositionalEmbeddings(nn.Module):
    def __init__(
        self,
        max_position_embeddings: int = 512,
        hidden_size: int = 768,
    ):
        super().__init__()
        assert isinstance(max_position_embeddings, int) and max_position_embeddings >= 1
        self.max_position_embeddings = max_position_embeddings
        assert isinstance(hidden_size, int) and hidden_size >= 1
        self.hidden_size = hidden_size

        self.embedder = nn.Embedding(self.max_position_embeddings, self.hidden_size)

    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 3  # (b s c)
        pe = self.embedder(torch.arange(x.shape[1], device=x.device)).repeat(
            x.shape[0], 1, 1
        )
        assert pe.shape == x.shape
        return pe


class AddGaussianNoise(nn.Module):
    def __init__(self, strength: float = 0.1):
        super().__init__()
        assert strength >= 0
        self.strength = strength

    def forward(self, x: torch.Tensor):
        noise = torch.normal(
            mean=torch.zeros_like(x, device=x.device, requires_grad=False) + x.mean(),
            std=torch.zeros_like(x, device=x.device, requires_grad=False) + x.std(),
        )
        noise = noise * self.strength
        return x + noise


class SATEER(EEGClassificationModel):
    def __init__(
        self,
        mels: int = 8,
        mel_window_size: Union[int, float] = 0.2,
        mel_window_stride: Union[int, float] = 0.25,
        users_embeddings: bool = False,
        num_users: Optional[int] = None,
        encoder_only: bool = False,
        hidden_size: int = 512,
        num_encoders: int = 4,
        num_decoders: int = 4,
        num_attention_heads: int = 8,
        positional_embedding_type: str = "sinusoidal",
        max_position_embeddings: int = 2048,
        dropout_p: Union[int, float] = 0.2,
        data_augmentation: bool = False,
        shifting: bool = False,
        cropping: bool = False,
        flipping: bool = False,
        noise_strength: Union[int, float] = 0.0,
        spectrogram_time_masking_perc: Union[int, float] = 0.05,
        spectrogram_frequency_masking_perc: Union[int, float] = 0.05,
        learning_rate: float = 1e-4,
        device: Optional[str] = None,
        **kwargs,
    ):
        super(SATEER, self).__init__(**kwargs)

        # metas
        self.in_channels = self.num_channels
        self.sampling_rate = self.eeg_sampling_rate

        self.labels_classes = [2 for _ in self.labels]
        assert isinstance(users_embeddings, bool)
        self.users_embeddings: bool = users_embeddings
        if self.users_embeddings:
            self.num_users: int = num_users
            self.users_dict: Dict[str, int] = {}

        # preprocessing
        assert (
            isinstance(mels, int) and mels >= 1
        ), f"the spectrogram must contain at least one mel bank"
        self.mels = self.n_mels
        self.mel_window_size = self.eeg_windows_size
        self.mel_window_stride = self.eeg_windows_stride

        # model architecture
        assert isinstance(encoder_only, bool)
        self.encoder_only = encoder_only
        assert isinstance(hidden_size, int) and hidden_size >= 1
        self.hidden_size = hidden_size
        assert isinstance(num_encoders, int) and num_encoders >= 1
        self.num_encoders: int = num_encoders
        if self.encoder_only is False:
            assert isinstance(num_decoders, int) and num_decoders >= 1
            self.num_decoders = num_decoders
        assert isinstance(num_attention_heads, int) and num_attention_heads >= 1
        self.num_attention_heads = num_attention_heads
        assert isinstance(
            positional_embedding_type, str
        ) and positional_embedding_type in {"sinusoidal", "learned"}
        self.positional_embedding_type = positional_embedding_type
        assert isinstance(max_position_embeddings, int) and max_position_embeddings >= 1
        self.max_position_embeddings = max_position_embeddings
        assert 0 <= dropout_p < 1
        self.dropout_p = dropout_p

        # regularization
        assert isinstance(data_augmentation, bool)
        self.data_augmentation = data_augmentation
        if self.data_augmentation is True:
            assert isinstance(shifting, bool)
            self.shifting = shifting
            assert isinstance(cropping, bool)
            self.cropping = cropping
            assert isinstance(flipping, bool)
            self.flipping = flipping
            assert noise_strength >= 0
            self.noise_strength = noise_strength
            assert 0 <= spectrogram_time_masking_perc < 1
            self.spectrogram_time_masking_perc = spectrogram_time_masking_perc
            assert 0 <= spectrogram_frequency_masking_perc < 1
            self.spectrogram_frequency_masking_perc = spectrogram_frequency_masking_perc

        # optimization
        assert isinstance(learning_rate, float) and learning_rate > 0
        self.learning_rate = self.lr = learning_rate
        assert device is None or device in {"cuda", "cpu"}
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.special_tokens_vocab: Dict[str, int] = {}
        self.special_tokens_vocab["ues"] = len(self.special_tokens_vocab)
        if len(self.special_tokens_vocab) > 0:
            self.special_tokens_embedder = nn.Embedding(
                len(self.special_tokens_vocab), self.hidden_size
            )
        if self.users_embeddings:
            # self.users_embedder = GetUserEmbeddings(hidden_size=self.hidden_size)
            self.users_embedder = nn.Embedding(self.num_users, self.hidden_size)
            self.token_type_embedder = nn.Embedding(3, self.hidden_size)
        if self.positional_embedding_type == "sinusoidal":
            self.position_embedder_spectrogram = GetSinusoidalPositionalEmbeddings(
                max_position_embeddings=self.max_position_embeddings,
            )
            self.position_embedder = GetSinusoidalPositionalEmbeddings(
                max_position_embeddings=self.max_position_embeddings,
            )
        elif self.positional_embedding_type == "learned":
            self.position_embedder_spectrogram = GetLearnedPositionalEmbeddings(
                max_position_embeddings=self.max_position_embeddings,
                hidden_size=self.in_channels * self.mels,
            )
            self.position_embedder = GetLearnedPositionalEmbeddings(
                max_position_embeddings=self.max_position_embeddings,
                hidden_size=self.hidden_size,
            )
        if self.encoder_only is False:
            self.labels_embedder = nn.Embedding(len(self.labels), self.hidden_size)

        self.merge_mels = nn.Sequential(
            Rearrange("b s c m -> b c m s"),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.hidden_size,
                kernel_size=(self.mels, 1),
                stride=1,
                padding=0,
            ),
            Rearrange("b c m s -> b s (c m)"),
            # nn.AdaptiveMaxPool2d(output_size=(self.hidden_size, 1)),
            # Rearrange("b s c m -> b s (c m)"),
            # nn.Linear(self.in_channels * self.mels, self.hidden_size),
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                batch_first=True,
                d_model=self.hidden_size,
                dim_feedforward=self.hidden_size * 4,
                dropout=self.dropout_p,
                activation=F.selu,
                nhead=self.num_attention_heads,
            ),
            num_layers=num_encoders,
        )

        if self.encoder_only is False:
            self.decoder = nn.TransformerDecoder(
                decoder_layer=nn.TransformerDecoderLayer(
                    batch_first=True,
                    d_model=self.hidden_size,
                    dim_feedforward=self.hidden_size * 4,
                    dropout=self.dropout_p,
                    activation=F.selu,
                    nhead=self.num_attention_heads,
                ),
                num_layers=num_decoders,
            )
            # replaces dropouts with alpha-dropout in the decoder
            for _, module in self.decoder.layers.named_children():
                for attr_str in dir(module):
                    target_attr = getattr(module, attr_str)
                    if type(target_attr) == nn.Dropout:
                        setattr(module, attr_str, nn.AlphaDropout(self.dropout_p))

        self.classification = nn.ModuleList()
        for i_label, label in enumerate(self.labels):
            self.classification.add_module(
                label,
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "linear1",
                                nn.Linear(
                                    in_features=self.hidden_size,
                                    out_features=self.labels_classes[i_label],
                                ),
                            ),
                            # ("linear1", nn.Linear(in_features=self.hidden_size,
                            #                       out_features=self.hidden_size * 4)),
                            # ("activation", nn.SELU()),
                            # ("dropout", nn.AlphaDropout(p=self.dropout_p)),
                            # ("linear2", nn.Linear(in_features=self.hidden_size * 4,
                            #                       out_features=self.labels_classes[i_label])),
                        ]
                    )
                ),
            )

        self.float()
        # self.to(device)

        self.cls_head = nn.Linear(self.h_dim, self.num_labels)
        if self.predict_ids:
            self.ids_head = nn.Linear(self.h_dim, len(self.id2int))
        self.save_hyperparameters()

    def forward(
        self,
        wf: torch.Tensor,
        mel_spec: torch.Tensor,
        ids: Optional[Union[int, str, List[Union[int, str]], np.ndarray]] = None,
        **kwargs
    ):
        outs = {}
        # optimizer = self.optimizers()
        # optimizer.optimizer.param_groups[0]["lr"] = 0.9
        # ensures that the inputs are well-defined
        input_eegs = einops.rearrange(wf, "b c s -> b s c")
        assert input_eegs.shape[-1] == self.in_channels, f"input_eegs.shape[-1] != self.in_channels: {input_eegs.shape} != {self.in_channels}"
        assert len(input_eegs.shape) in {2, 3}
        if ids is not None:
            if isinstance(ids, int) or isinstance(ids, str):
                ids = [ids]
            elif isinstance(ids, list):
                assert all([isinstance(id, int) or isinstance(id, str) for id in ids])
            # if isinstance(ids, list):
            # else:
            #     raise TypeError(f"ids must be a string, an integer or a list of such, not {type(ids)}")
            assert len(input_eegs) == len(
                ids
            ), f"length between eegs and ids mismatch: {len(input_eegs)} != {len(ids)}"

        # makes a fresh copy of the input to avoid errors
        eegs = input_eegs.clone()  # (b s c) or (s c)

        # eventually adds a batch dimension
        is_batched = True if len(eegs.shape) == 3 else False
        if not is_batched:
            eegs = einops.rearrange(eegs, "s c -> () s c")

        # eventually adds data augmentation
        if self.training is True and self.data_augmentation is True:
            # with profiler.record_function("data augmentation (eegs)"):
            if self.shifting is True:
                for i_batch in range(eegs.shape[0]):
                    shift_direction = (
                        "left" if torch.rand(1, device=eegs.device) <= 0.5 else "right"
                    )
                    shift_amount = int(
                        torch.rand(1, device=eegs.device) * 0.25 * eegs.shape[1]
                    )
                    assert shift_amount < eegs.shape[1]
                    if shift_amount > 0:
                        if shift_direction == "left":
                            eegs[i_batch] = torch.roll(
                                eegs[i_batch], shifts=shift_amount, dims=0
                            )
                            eegs[i_batch, :shift_amount] = 0
                        else:
                            eegs[i_batch] = torch.roll(
                                eegs[i_batch], shifts=-shift_amount, dims=0
                            )
                            eegs[i_batch, -shift_amount:] = 0
            if self.cropping is True:
                crop_amount = int(
                    eegs.shape[1] - torch.rand(1, device=eegs.device) * 0.25 * eegs.shape[1]
                )
                if crop_amount > 0:
                    for i_batch in range(eegs.shape[0]):
                        crop_start = int(
                            torch.rand(1, device=eegs.device)
                            * (eegs.shape[1] - crop_amount)
                        )
                        assert crop_start + crop_amount < eegs.shape[1]
                        eegs[i_batch, :crop_amount] = eegs[
                            i_batch, crop_start : crop_start + crop_amount
                        ].clone()
                eegs = eegs[:, :crop_amount]
            if self.flipping is True:
                for i_batch, batch in enumerate(eegs):
                    if torch.rand(1, device=eegs.device) <= 0.25:
                        eegs[i_batch] = torch.flip(eegs[i_batch], dims=[0])
            if self.noise_strength > 0:
                eegs = AddGaussianNoise(strength=self.noise_strength)(eegs)

        # converts the eegs to a spectrogram
        # with profiler.record_function("spectrogram"):
        spectrogram = einops.rearrange(mel_spec, "b c m s -> b s c m")   # (b s c m)
        assert spectrogram.shape[2:] == (self.num_channels, self.mels), f"expected {spectrogram.shape[2:]} == {(self.num_channels, self.mels)}" 
        # MelSpectrogram.plot_mel_spectrogram(spectrogram[0])
        if self.training is True and self.data_augmentation is True:
            # MelSpectrogram.plot_mel_spectrogram(spectrogram[0])
            # with profiler.record_function("data augmentation (spectrogram)"):
            if self.spectrogram_time_masking_perc > 0:
                for i_batch in range(len(spectrogram)):
                    mask_amount = int(
                        random.random()
                        * self.spectrogram_time_masking_perc
                        * spectrogram.shape[1]
                    )
                    if mask_amount > 0:
                        masked_indices = torch.randperm(
                            spectrogram.shape[1], device=spectrogram.device
                        )[:mask_amount]
                        spectrogram[i_batch, masked_indices] = 0
            if self.spectrogram_frequency_masking_perc > 0:
                for i_batch in range(len(spectrogram)):
                    mask_amount = int(
                        random.random()
                        * self.spectrogram_frequency_masking_perc
                        * spectrogram.shape[-1]
                    )
                    if mask_amount > 0:
                        masked_indices = torch.randperm(
                            spectrogram.shape[-1], device=spectrogram.device
                        )[:mask_amount]
                        spectrogram[i_batch, :, :, masked_indices] = 0

        # prepares the spectrogram for the encoder
        # with profiler.record_function("preparation"):
        x = self.merge_mels(spectrogram)  # (b s c)
        assert len(x.shape) == 3, f"invalid number of dimensions ({x.shape} must be long 3)"
        assert (
            x.shape[-1] == self.hidden_size
        ), f"invalid hidden size after merging ({x.shape[-1]} != {self.hidden_size})"

        # pass the spectrogram through the encoder
        # with profiler.record_function("encoder"):
        # eventually adds positional embeddings and type embeddings
        if self.users_embeddings and (ids is not None):
            # with profiler.record_function("user embeddings"):
            # eventually adds a new user to the dict
            for user_id in ids:
                if user_id not in self.users_dict:
                    self.users_dict[user_id] = len(self.users_dict)
            # generates an embedding for each users
            users_embeddings = self.users_embedder(
                torch.as_tensor(
                    [self.users_dict[user_id] for user_id in ids], device=self.device
                )
            )  # (b c)
            # concatenates the embeddings to the eeg
            x = torch.cat(
                [
                    users_embeddings.unsqueeze(1),
                    self.special_tokens_embedder(
                        torch.as_tensor(
                            [self.special_tokens_vocab["ues"]], device=self.device
                        )
                    ).repeat(x.shape[0], 1, 1),
                    x,
                ],
                dim=1,
            )
            x = x + self.token_type_embedder(
                torch.as_tensor(
                    [0, 1, *[2 for _ in range(x.shape[1] - 2)]], device=self.device
                )
            ).repeat(x.shape[0], 1, 1)
        # adds the positional embeddings
        x = x + self.position_embedder(x)
        # encoder pass
        x_encoded = self.encoder(x)  # (b s d)
        if self.users_embeddings and (ids is not None):
            x_encoded = x_encoded[:, 2:]
        assert (
            len(x_encoded.shape) == 3
        ), f"invalid number of dimensions ({x_encoded.shape} must be long 3)"
        assert (
            x_encoded.shape[-1] == self.hidden_size
        ), f"invalid hidden size after encoder ({x_encoded.shape[-1]} != {self.hidden_size})"

        # eventually pass the encoded spectrogram to the decoder
        if self.encoder_only is False:
            # with profiler.record_function("decoder"):
            # prepares the labels tensor
            label_tokens = self.labels_embedder(
                torch.as_tensor(list(range(len(self.labels))), device=x_encoded.device)
            ).repeat(
                x_encoded.shape[0], 1, 1
            )  # (b l d)
            # adds the positional embeddings
            label_tokens = label_tokens + self.position_embedder(label_tokens)  # (b l d)
            # decoder pass
            x_decoded = self.decoder(label_tokens, x_encoded)  # (b l d)
            assert (
                len(x_decoded.shape) == 3
            ), f"invalid number of dimensions ({x_decoded.shape} must be long 3)"
            assert (
                x_decoded.shape[-1] == self.hidden_size
            ), f"invalid hidden size after merging ({x_decoded.shape[-1]} != {self.hidden_size})"

        # makes the predictions using the encoded or decoded data
        # with profiler.record_function("predictions"):
        if self.encoder_only is True:
            # classification
            features = x_encoded[:, 0, :]
        else:
            features = torch.mean(x_encoded, dim=1)
        outs["features"] = features
        # outs["cls_logits"] = self.cls_head(features)
        # if self.predict_ids:
        #     outs["ids_logits"] = self.ids_head(features)
        return outs
