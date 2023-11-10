import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio


class SISDRLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, gamma=0.5, device='cuda:0'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sisdr = ScaleInvariantSignalDistortionRatio().to(device)

    def forward(self, short, middle, long, targets, speaker_ids, logits, train=True, **kwargs):
        sisdr_short = self.sisdr(short.squeeze(1), targets.squeeze(1))
        sisdr_middle = self.sisdr(middle.squeeze(1), targets.squeeze(1))
        sisdr_long = self.sisdr(long.squeeze(1), targets.squeeze(1))

        sisdr_loss = (1 - self.alpha - self.beta) * sisdr_short.sum() + self.alpha * sisdr_middle.sum() + self.beta * sisdr_long.sum()
        sisdr_loss = -sisdr_loss.mean()

        if not train:
            return sisdr_loss
        ce_loss = F.cross_entropy(logits, speaker_ids)
        return sisdr_loss + self.gamma * ce_loss

