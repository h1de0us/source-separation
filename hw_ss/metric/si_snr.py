from typing import List

import torch
from torch import Tensor
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio as SISNR

from hw_ss.base.base_metric import BaseMetric

class SiSnrMetric(BaseMetric):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.sisnr = SISNR()

    def __call__(self, **batch):
        preds = batch["short"]
        target = batch["target_audio"]
        min_len = min(target.shape[-1], preds.shape[-1])
        return self.sisnr(preds[:, :, :min_len], target[:, :, :min_len])