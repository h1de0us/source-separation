import torch
from torch import Tensor

from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

class SiSnrWrapper(ScaleInvariantSignalNoiseRatio):
    def __init__(self, ):
        super(SiSnrWrapper, self).__init__()


    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        return -super().forward(preds, target)
        