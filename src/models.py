import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from scipy.signal import savgol_filter
import numpy as np

class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128,
        z_dim=512,
        flag=0
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.hid_dim = hid_dim
        self.z_dim = z_dim
        self.flag = flag

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            #ConvBlock(hid_dim, hid_dim),
        )

        self.z_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, z_dim),
        )

        self.z_block = nn.Sequential(
              nn.Linear(z_dim, hid_dim),
              nn.ReLU(),
              nn.Linear(hid_dim, num_classes)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        #平滑化
        X_cpu = X.cpu()
        X = savgol_filter(X_cpu.numpy(), window_length=10, polyorder=2) #numpy

        #minとmaxでスケーリング
        X = torch.from_numpy((X - np.min(X, axis=2, keepdims=True)) / (np.max(X, axis=2, keepdims=True) - np.min(X, axis=2, keepdims=True))).cuda()
        
        #ベースライン補正
        mean_in_start = torch.mean(X[:,:,:25], dim=2, keepdim=True)
        X = X - mean_in_start

        X = self.blocks(X)
        z = self.z_head(X)

        if self.flag == 1:
          return z
        else:
          output = self.z_block(z)
          return output


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.3,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        #self.conv2 = nn.Conv1d(z_dim, out_dim, kernel_size, padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X)
        z = F.gelu(self.batchnorm1(X))

        return self.dropout(z)