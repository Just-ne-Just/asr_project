from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from hw_asr.metric.utils import calc_si_sdr
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

class SISDRMetric(BaseMetric):
    def __init__(self, device='cuda:0', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sisdr = ScaleInvariantSignalDistortionRatio().to(device)

    def __call__(self, short, targets, **kwargs):
        targets = targets.squeeze(1)
        short = short.squeeze(1)
        sisdr = self.sisdr(short, targets)
        return sisdr.mean().item()