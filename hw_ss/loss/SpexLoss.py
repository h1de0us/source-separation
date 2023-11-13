import torch
from torch import Tensor
from torch import nn

from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from torch.nn import CrossEntropyLoss


class SpexLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, gamma=0.5) -> None:
        super().__init__()
        self.si_snr = ScaleInvariantSignalNoiseRatio()
        self.cross_entropy = CrossEntropyLoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, **batch) -> Tensor:
        logits = batch["logits"]
        speaker_index = batch["librispeech_speaker_index"]
        ce_loss = self.cross_entropy(logits, speaker_index)

        short, middle, long = batch["short"], batch["middle"], batch["long"]
        target = batch["target_audio"]
        min_len = min(short.shape[-1], target.shape[-1])
        short = short[:, :, :min_len]
        trimmed_target = target[:, :, :min_len]

        si_snr_loss = (1 - self.alpha - self.beta) * self.si_snr(short, trimmed_target)

        min_len = min(middle.shape[-1], target.shape[-1])
        middle = middle[:, :, :min_len]
        trimmed_target = target[:, :, :min_len]

        si_snr_loss += self.alpha * self.si_snr(middle, trimmed_target) 
        
        min_len = min(long.shape[-1], target.shape[-1])
        long = long[:, :, :min_len]
        trimmed_target = target[:, :, :min_len]

        si_snr_loss += self.beta * self.si_snr(long, trimmed_target)
        
        return self.gamma * ce_loss - si_snr_loss

        
