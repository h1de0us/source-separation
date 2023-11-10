from hw_ss.base import BaseModel

import torch
from torch import nn
from torch.nn import Conv1d, BatchNorm1d, MaxPool1d, PReLU, LayerNorm, AvgPool1d
from torch.nn import functional as F

class SpeechEncoder(nn.Module):
    def __init__(self, L1, L2, L3, out_channels) -> None:
        super(SpeechEncoder, self).__init__()
        # L1(short), L2(middle) and L3(long) are the filter length 
        # of each filter to capture different temporal resolution in the 1-D CNN.

        assert L1 <= L2 <= L3

        self.L1_short = Conv1d(in_channels=1, 
                               out_channels=out_channels,
                               kernel_size=L1,
                               stride = L1 // 2)
        self.L2_middle = Conv1d(in_channels=1,
                                out_channels=out_channels,
                                kernel_size=L2,
                                stride = L1 // 2)
        self.L3_long = Conv1d(in_channels=1,
                                out_channels=out_channels,
                                kernel_size=L3,
                                stride = L1 // 2)


    def forward(self, x):
        short, middle, long = self.L1_short(x), self.L2_middle(x), self.L3_long(x)
        return torch.cat([short, middle, long], 1) # TODO: check dimensions


class GlobalLayerNorm(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(GlobalLayerNorm, self).__init__()


    def forward(self, x):
        pass
    # TODO !!!!


# class DDSepConvolution(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super(DDSepConvolution, self).__init__()


#     def forward(self, x):
#         pass


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_first=False, kernel_size=3) -> None:
        super(TCNBlock, self).__init__()
        self.is_first = is_first
        self.conv_in = Conv1d(in_channels,
                            out_channels,
                            kernel_size=1)
        self.prelu1 = PReLU(out_channels)
        self.global_layer_norm1 = GlobalLayerNorm() # TODO: some params
        self.de_cnn = Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=out_channels,
            padding=(dilation * (kernel_size - 1)) // 2,
            dilation=dilation,
            bias=True)
        # “De-CNN” indicates a dilated depth-wise separable convolution.
        
        self.prelu2 = PReLU(out_channels)
        self.global_layer_norm2 = GlobalLayerNorm() # TODO: some params
        self.conv_out = Conv1d(out_channels,
                            out_channels,
                            kernel_size=1)
        
        # TODO: think about is_first
        

    def forward(self, x):
        # TODO: think about is_first
        if self.is_first:
            pass
        outputs = self.conv_in(x)
        outputs = self.prelu1(outputs)
        outputs = self.global_layer_norm1(outputs)
        outputs = self.de_cnn(outputs)
        outputs = self.prelu2(outputs)
        outputs = self.global_layer_norm2(outputs)
        outputs = self.conv_out(outputs)
        outputs = outputs + x
        return outputs



class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(ResNetBlock, self).__init__()
        self.conv1 = Conv1d(in_channels, 
                            out_channels,
                            kernel_size=1)
        self.batch_norm1 = BatchNorm1d(out_channels)
        self.prelu1 = PReLU(out_channels)
        self.conv2 = Conv1d(out_channels,
                            out_channels,
                            kernel_size=1)
        self.batch_norm2 = BatchNorm1d(out_channels)
        self.prelu2 = PReLU(out_channels)
        self.pooling = MaxPool1d(kernel_size=(1, 3)) # see paper for more info
        # the time series of the representations are reduced by 3 times

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.batch_norm1(outputs)
        outputs = self.prelu1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.batch_norm2(outputs)
        outputs = outputs + x
        outputs = self.prelu2(outputs)
        outputs = self.pooling(outputs)
        return outputs


class SpeakerEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, n_resblocks) -> None:
        '''
        Speaker encoder is designed to extract speaker embedding 
        of the target speaker from the reference speech. 
        In practice, we employ a 1-D CNN on the embedding coefficients X from the reference speech, 
        followed by residual network (ResNet) blocks with a number of NR. 
        Then a 1-D CNN is used to project the representations into 
        a fixed dimensional utterance-level speaker embedding v 
        together with a mean pooling operation. 
        We have v = g(X), where g(·) represents the speaker encoder.
        '''
        super(SpeakerEncoder, self).__init__()
        self.norm = LayerNorm() # TODO: is it really layer norm?
        self.conv_in = Conv1d(in_channels,
                              out_channels,
                              kernel_size=1)
        self.resblocks = nn.ModuleList(
            ResNetBlock(in_channels, out_channels) for _ in range(n_resblocks)
        )
        self.conv_out = Conv1d(out_channels,
                               out_channels,
                               kernel_size=1)
        self.pooling = AvgPool1d()

    def forward(self, x):
        embeds = self.norm(x)
        embeds = self.conv_in(embeds)
        for layer in self.resblocks:
            embeds = layer(embeds)
        embeds = self.conv_out(embeds)
        embeds = self.pooling(embeds)
        return embeds
    
# TODO: class for speech decoder
class SpeechDecoder(nn.Module):
    def __init__(self) -> None:
        super(SpeechDecoder, self).__init__()
        pass

    def forward(self, x):
        pass
        

class SpeakerExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, n_tcns=4) -> None:
        super(SpeakerExtractor, self).__init__()
        '''
        Speaker extractor is designed to estimate masks M_i for the target speaker in each scale i = 1, 2, 3, 
        conditioned on the embedding coefficients Y and the speaker embedding v. 
        Similar to Conv-TasNet [13], our speaker extractor repeats a stack of
        temporal convolutional network (TCN) blocks with a number of B for R times.
        In this work, we keep B=8 and R=4 same as in SpEx [14]. 
        In each stack, the TCN has a exponential growth dilation factor 2b (b ∈ {0, · · · , B - 1}) 
        in the dilated depth-wise separable convolution (De-CNN).
        The first TCN block in each stack takes the speaker embedding v and the learned representations over the mixture speech. 
        The speaker embedding v is repeated to be concatenated to the feature dimension of the representation.
        Once the masks M_i in various scales are estimated, the modulated responses S_i are obtained 
        by element-wise multiplication of the masks M_i and the embedding coefficients Y_i. 
        We then reconstruct the modulated responses S_i into time-domain signals s_i at multiple scales with 
        the multi-scale speech decoder as follows:
        si = d(Mi ⊗ Yi) = d(f(Y, v) ⊗ Yi) (3) (operation for element-wise multiplication)
        f(·) and d(·) are the speaker extractor to estimate the mask and the speech decoder to reconstruct the signal, respectively.
        '''
        self.norm = LayerNorm()
        self.conv_in = Conv1d(in_channels,
                              out_channels,
                              kernel_size=1)
        self.tcns = nn.ModuleList(
            TCNBlock(out_channels, out_channels, is_first=(i == 0)) for i in range(n_tcns)
        )
        self.mask1 = Conv1d(...) # TODO: channels
        self.mask2 = Conv1d(...) # TODO: channels
        self.mask3 = Conv1d(...) # TODO: channels

        self.relu1 = F.relu()
        self.relu2 = F.relu()
        self.relu3 = F.relu()

        # TODO:
        # point-wise multiplication
        # parameters for three decoders
        


    def forward(self, speech, speaker_embeds):
        # TODO: proper forward pass, return THREE values
        pass



class SpexPlus(BaseModel):
    def __init__(self, 
                 L1: int,
                 L2: int,
                 L3: int,
                 speech_enc_out_channels: int,
                 spk_in_channels: int, 
                 spk_out_channels: int, 
                 n_resblocks: int, 
                 spex_in_channels: int,
                 spex_out_channels: int,
                 n_tcns: int) -> None:
        super(SpexPlus, self).__init__()
        self.speech_encoder = SpeechEncoder(L1, L2, L3, speech_enc_out_channels)
        self.speaker_encoder = SpeakerEncoder(spk_in_channels, spk_out_channels, n_resblocks)
        self.speaker_extractor = SpeakerExtractor(in_channels=spex_in_channels,
                                                  out_channels=spex_out_channels,
                                                  n_tcns=n_tcns)
        
        self.speech_decoder_short = SpeechDecoder()
        self.speech_decoder_middle = SpeechDecoder()
        self.speech_decoder_long = SpeechDecoder()

        


    def forward(self, reference, mixture) -> torch.Tensor:
        # x -- reference speech
        # y -- mixture speech
        x, y = self.speech_encoder(reference), self.speech_encoder(mixture)
        speaker_embeds = self.speaker_encoder(x)
        short, middle, long = self.speaker_extractor(y, speaker_embeds)
        outputs = (self.speech_decoder_short(short), 
                   self.speech_decoder_middle(middle),
                   self.speech_decoder_long(long))
        return outputs
