from loguru import logger
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from models.base_model import EEGClassificationModel


class LightweightConv1d(nn.Module):

    def __init__(
        self,
        in_channels,
        num_heads=1,
        depth_multiplier=1,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True,
        weight_softmax=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_heads = num_heads
        self.padding = padding
        self.weight_softmax = weight_softmax
        self.weight = nn.Parameter(
            torch.Tensor(num_heads * depth_multiplier, 1, kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_heads * depth_multiplier))
        else:
            self.bias = None

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, inp):
        B, C, T = inp.size()
        H = self.num_heads

        weight = self.weight
        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)

        # input = input.view(-1, H, T)
        inp = rearrange(inp, "b (h c) t ->(b c) h t", h=H)
        if self.bias is None:
            output = F.conv1d(
                inp,
                weight,
                stride=self.stride,
                padding=self.padding,
                groups=self.num_heads,
            )
        else:
            output = F.conv1d(
                inp,
                weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                groups=self.num_heads,
            )
        output = rearrange(output, "(b c) h t ->b (h c) t", b=B)

        return output


class Mixer1D(nn.Module):
    def __init__(self, dim, kernel_sizes=[50, 100, 250]):
        super().__init__()
        self.var_layers = nn.ModuleList()
        self.L = len(kernel_sizes)
        for k in kernel_sizes:
            self.var_layers.append(
                nn.Sequential(
                    VarPool1D(kernel_size=k, stride=int(k / 2)),
                    nn.Flatten(start_dim=1),
                )
            )

    def forward(self, x):
        B, d, L = x.shape
        x_split = torch.split(x, d // self.L, dim=1)
        out = []
        for i in range(len(x_split)):
            x = self.var_layers[i](x_split[i])
            out.append(x)
        y = torch.concat(out, dim=1)
        return y


class VarPool1D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Calculate the size of the result tensor after pooling

        # Compute the mean of the squares (E[x^2])
        mean_of_squares = F.avg_pool1d(
            x**2, self.kernel_size, self.stride, self.padding
        )

        # Compute the square of the mean (E[x])^2
        square_of_mean = (
            F.avg_pool1d(x, self.kernel_size, self.stride, self.padding) ** 2
        )

        # Compute the variance: Var[X] = E[X^2] - (E[X])^2
        variance = mean_of_squares - square_of_mean

        return variance


class VarMaxPool1D(nn.Module):
    def __init__(self, T, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride
        self.padding = padding

    def forward(self, x):
        mean_of_squares = F.avg_pool1d(
            x**2, self.kernel_size, self.stride, self.padding
        )
        # Compute the square of the mean (E[x])^2
        square_of_mean = (
            F.avg_pool1d(x, self.kernel_size, self.stride, self.padding) ** 2
        )

        # Compute the variance: Var[X] = E[X^2] - (E[X])^2
        variance = mean_of_squares - square_of_mean
        # out = self.time_agg(variance)
        out = F.avg_pool1d(variance, variance.shape[-1])

        return out


class SSA(nn.Module):
    # Spatial-Spectral Attention
    def __init__(self, T, num_channels, epsilon=1e-5, mode="var", after_relu=False):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

        self.GP = VarMaxPool1D(T, 64)

    def forward(self, x):
        B, C, T = x.shape

        if self.mode == "l2":
            embedding = (x.pow(2).sum((2), keepdim=True) + self.epsilon).pow(0.5)
            norm = self.gamma / (
                embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon
            ).pow(0.5)

        elif self.mode == "l1":
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2), keepdim=True)
            norm = self.gamma / (
                torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon
            )

        elif self.mode == "var":

            embedding = (self.GP(x) + self.epsilon).pow(0.5) * self.alpha
            norm = (self.gamma) / (
                embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon
            ).pow(0.5)

        gate = 1 + torch.tanh(embedding * norm + self.beta)

        return x * gate, gate


class Efficient_Encoder(nn.Module):

    def __init__(
        self,
        samples,
        chans,
        F1=16,
        F2=36,
        time_kernel1=32,
        pool_kernels=[16, 32, 64],
    ):
        super().__init__()

        self.time_conv = LightweightConv1d(
            in_channels=chans,
            num_heads=1,
            depth_multiplier=F1,
            kernel_size=time_kernel1,
            stride=1,
            padding="same",
            bias=True,
            weight_softmax=False,
        )
        self.ssa = SSA(samples, chans * F1)

        self.chanConv = nn.Sequential(
            nn.Conv1d(
                chans * F1,
                F2,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm1d(F2),
            nn.ELU(),
        )

        self.mixer = Mixer1D(dim=F2, kernel_sizes=pool_kernels)

    def forward(self, x):

        x = self.time_conv(x)
        x, _ = self.ssa(x)
        x_chan = self.chanConv(x)

        feature = self.mixer(x_chan)

        return feature


class EDPNet(EEGClassificationModel):

    def __init__(
        self, F1=8, F2=48, time_kernel1=32, pool_kernels=[16, 32, 64], **kwargs
    ):
        super(EDPNet, self).__init__(**kwargs)

        self.encoder = Efficient_Encoder(
            samples=self.eeg_samples,
            chans=self.num_channels,
            F1=F1,
            F2=F2,
            time_kernel1=time_kernel1,
            pool_kernels=pool_kernels,
        )
        self.features = None

        x = torch.ones((1, self.num_channels, self.eeg_samples))
        out = self.encoder(x)
        self.h_dim = out.shape[-1]

        # *Inter-class Separation Prototype(ISP)
        self.isp_cls = nn.Parameter(
            torch.randn(self.num_labels, self.h_dim), requires_grad=True
        )
        nn.init.kaiming_normal_(self.isp_cls)
        if self.predict_ids:
            self.isp_ids = nn.Parameter(
                torch.randn(len(self.id2int), self.h_dim), requires_grad=True
            )
            nn.init.kaiming_normal_(self.isp_ids)
        self.save_hyperparameters()

    def get_features(self):
        if self.features is not None:
            return self.features
        else:
            raise RuntimeError("No features available. Run forward() first.")

    def forward(self, wf, **kwargs):
        outs = {}

        # feature extraction
        features = self.encoder(wf)
        self.features = features

        # classification
        self.isp_cls.data = torch.renorm(self.isp_cls.data, p=2, dim=0, maxnorm=1)
        outs["cls_logits"] = torch.einsum("bd,cd->bc", features, self.isp_cls)
        if self.predict_ids:
            self.isp_ids.data = torch.renorm(
                self.isp_ids.data, p=2, dim=0, maxnorm=1
            )
            outs["ids_logits"] = torch.einsum("bd,cd->bc", features, self.isp_ids)
        return outs
