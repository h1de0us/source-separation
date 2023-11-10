from typing import List

import torch
from torch import Tensor
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio as SISNR

from hw_ss.base.base_metric import BaseMetric

class SiSnrMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sisnr = SISNR(**kwargs)

    def __call__(self, preds: Tensor, target: Tensor, text: List[str], **kwargs):
        return self.sisnr(preds, target)