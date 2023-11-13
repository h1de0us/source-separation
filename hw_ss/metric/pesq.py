from typing import List

import torch
from torch import Tensor

from hw_ss.base.base_metric import BaseMetric
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ

class PESQMetric(BaseMetric):
    def __init__(self, name, fs, mode, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.pesq = PESQ(fs=fs, mode=mode)

    def __call__(self, **batch):
        target = batch["target_audio"]
        preds = batch["short"]

        min_len = min(target.shape[-1], preds.shape[-1])
        preds, target = preds[:, :, :min_len], target[:, :, :min_len]
        preds = preds.squeeze(1) # remove dimension for channels
        target = target.squeeze(1) # remove dimension for channels
        return self.pesq(preds, target)
        