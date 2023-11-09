from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from hw_asr.metric.utils import calc_si_sdr
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

class SISDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sisdr = ScaleInvariantSignalDistortionRatio()

    def __call__(self, short, targets, **kwargs):
        targets = targets.squeeze(1)
        short = short.squeeze(1)
        self.sisdr = self.sisdr.to(short.device)
        sisdr = self.sisdr(short, targets)
        return sisdr.mean().item()