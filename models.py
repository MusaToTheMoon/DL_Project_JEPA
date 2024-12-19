from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch



# -----------------------------------------------------
# VICReg-style Loss
# -----------------------------------------------------
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n-1, n+1)[:,1:].flatten()

def compute_vicreg_loss(x, y, sim_weight=25.0, var_weight=25.0, cov_weight=1.0, eps=1e-4):
    B_feat, C, H, W = x.shape
    x_flat = x.reshape(B_feat, C*H*W)
    y_flat = y.reshape(B_feat, C*H*W)

    inv_loss = F.mse_loss(x_flat, y_flat)

    x_std = torch.sqrt(x_flat.var(dim=0) + eps)
    y_std = torch.sqrt(y_flat.var(dim=0) + eps)
    var_loss = torch.mean(F.relu(1 - x_std)) + torch.mean(F.relu(1 - y_std))

    x_mean = x_flat - x_flat.mean(dim=0, keepdim=True)
    y_mean = y_flat - y_flat.mean(dim=0, keepdim=True)

    x_cov = (x_mean.T @ x_mean) / (B_feat - 1)
    y_cov = (y_mean.T @ y_mean) / (B_feat - 1)

    x_cov_offdiag = off_diagonal(x_cov).pow_(2).sum() / (C*H*W)
    y_cov_offdiag = off_diagonal(y_cov).pow_(2).sum() / (C*H*W)
    cov_loss = x_cov_offdiag + y_cov_offdiag

    loss = sim_weight * inv_loss + var_weight * var_loss + cov_weight * cov_loss
    return loss

class Encoder(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, o):
        B, T, C, H, W = o.shape
        x = o.reshape(B*T, C, H, W).contiguous()
        s = self.net(x)
        s = s.reshape(B, T, 32, 8, 8)
        return s

class Predictor(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()

        self.conv1 = nn.Conv2d(34, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, s_prev, u_prev):
        B, Tm1, C, H, W = s_prev.shape
        u_prev = u_prev.unsqueeze(-1).unsqueeze(-1).expand(B, Tm1, 2, H, W)
        x = torch.cat([s_prev, u_prev], dim=2)
        x = x.reshape(B*Tm1, 34, 8, 8).contiguous()

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        residual = out
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out + residual)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = F.relu(out)
        out = out.reshape(B, Tm1, 32, 8, 8)
        return out

class NonRecurrentJEPA(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(dropout_rate=dropout_rate)
        self.predictor = Predictor(dropout_rate=dropout_rate)
        self.repr_dim = 32 * 8 * 8
        print("JEPA created")

    def forward(self, states, actions):
        B = states.shape[0]
        T = states.shape[1]

        s_enc = self.encoder(states)

        if T > 1:
            s_prev = s_enc[:, :-1]
            pred_s = self.predictor(s_prev, actions)

            s_all = torch.cat([s_enc[:,0:1], pred_s], dim=1)
        else:
            s_all = s_enc

        s_all_flat = s_all.reshape(B, T, self.repr_dim)
        return s_all_flat

    def compute_loss(self, states, actions):
        B, T = states.shape[0], states.shape[1]
        s_enc = self.encoder(states)
        s_prev = s_enc[:, :-1]
        pred_s = self.predictor(s_prev, actions)
        s_target = s_enc[:,1:]

        B_Tm1 = (B*(T-1)) if (T>1) else 0
        if B_Tm1 == 0:
            return torch.tensor(0.0, device=states.device)

        pred_s_flat = pred_s.reshape(B_Tm1, 32, 8, 8)
        s_target_flat = s_target.reshape(B_Tm1, 32, 8, 8)
        loss = compute_vicreg_loss(pred_s_flat, s_target_flat)
        return loss




def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)



class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output

