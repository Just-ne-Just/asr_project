from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

def audio_norm(audio):
    return 20 * audio / audio.norm(dim=-1, keepdim=True)

class PESQMetric(BaseMetric):
    def __init__(self, fs=16000, mode='wb', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesq = PerceptualEvaluationSpeechQuality(fs, mode)

    def __call__(self, short, targets, **kwargs):
        targets = targets.squeeze(1)
        short = short.squeeze(1)
        self.pesq = self.pesq.to(short.device)
        return self.pesq(audio_norm(short), targets).mean()