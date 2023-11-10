from typing import List

import torch
from torch import Tensor

from hw_ss.base.base_metric import BaseMetric
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ


# TODO: calculate metric properly
class PESQMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesq = PESQ(**kwargs)

    def __call__(self, preds: Tensor, target: Tensor, **kwargs):
        return self.pesq(preds, target)
        