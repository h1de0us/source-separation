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
        self.L1, self.L2, self.L3 = L1, L2, L3

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
        short = self.L1_short(x)
        short_len, x_len = short.shape[-1], x.shape[-1]
        middle_len = (short_len - 1) * (self.L1 // 2) + self.L2
        long_len = (short_len - 1) * (self.L1 // 2) + self.L3
        middle = self.L2_middle(F.pad(x, (0, middle_len - x_len), "constant", 0))
        long = self.L3_long(F.pad(x, (0, long_len - x_len), "constant", 0))
        # print('short, middle, long shapes:', short.shape, middle.shape, long.shape)
        return short, middle, long


class GlobalLayerNorm(nn.Module):
    def __init__(self) -> None:
        super(GlobalLayerNorm, self).__init__()

    def forward(self, x):
        mean = torch.mean(x, (1, 2))
        std = torch.std(x, (1, 2))
        return (x - mean) / (std + 1e-06)


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
        self.pooling = MaxPool1d(kernel_size=3)
        # the time series of the representations are reduced by 3 times

        self.downsample = in_channels != out_channels
        self.conv_downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # print(f'x.shape: {x.shape}')
        outputs = self.conv1(x)
        # print(f'x.shape after convolution: {outputs.shape}')
        outputs = self.batch_norm1(outputs)
        outputs = self.prelu1(outputs)
        outputs = self.conv2(outputs)
        # print(f'x.shape after second convolution: {outputs.shape}')
        outputs = self.batch_norm2(outputs)
        if self.downsample:
            outputs = outputs + self.conv_downsample(x)
        else:
            outputs = outputs + x
        outputs = self.prelu2(outputs)
        outputs = self.pooling(outputs)
        # print(f'x.shape after pooling: {outputs.shape}')
        return outputs


# a class to get embeddings for a speaker
class SpeakerEncoder(nn.Module):
    def __init__(self, N, O, P, speaker_embed_dim) -> None:
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
        self.norm = LayerNorm(3 * N)
        self.conv_in = Conv1d(3 * N, O, kernel_size=1)
        self.resblocks = nn.ModuleList([
            ResNetBlock(O, O),
            ResNetBlock(O, P),
            ResNetBlock(P, P)
        ])
        self.conv_out = Conv1d(P,
                               speaker_embed_dim,
                               kernel_size=1)

    def forward(self, short, middle, long):
        # print(short.shape, middle.shape, long.shape)
        x = torch.cat([short, middle, long], dim=1)
        embeds = x.transpose(1, 2)
        embeds = self.norm(embeds)
        embeds = embeds.transpose(1, 2)
        embeds = self.conv_in(embeds)
        for layer in self.resblocks:
            embeds = layer(embeds)
        embeds = self.conv_out(embeds)
        embeds = F.avg_pool1d(embeds, kernel_size=embeds.shape[-1])
        # print('embeds shape after avg pooling', embeds.shape)
        return embeds
    

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, speak_embed_dim, is_first=False, de_cnn_kernel_size=3, dilation=1) -> None:
        super(TCNBlock, self).__init__()
        self.is_first = is_first
        if self.is_first:
            self.conv_in = Conv1d(in_channels + speak_embed_dim, out_channels, kernel_size=1)
        else:
            self.conv_in = Conv1d(in_channels, out_channels, kernel_size=1)
            
        self.prelu1 = PReLU(out_channels)
        self.global_layer_norm1 = GlobalLayerNorm()
        self.de_cnn = Conv1d(
            out_channels,
            out_channels,
            kernel_size=de_cnn_kernel_size,
            groups=out_channels,
            padding=(dilation * (de_cnn_kernel_size - 1)) // 2,
            dilation=dilation,
            bias=True)
        # “De-CNN” indicates a dilated depth-wise separable convolution.
        # The dilated depth-wise convolution has a kernel size of 1 × Q, 
        # a number of P filters and a dilation factor of 2^(B−1). B is the number of TCN blocks in a stack.
        
        self.prelu2 = PReLU()
        self.global_layer_norm2 = GlobalLayerNorm()
        self.conv_out = Conv1d(out_channels,
                            in_channels,
                            kernel_size=1)
        

    def forward(self, x, speaker_embeds=None):
        outputs = x
        if self.is_first:
            assert speaker_embeds is not None, "speaker embeds is None for first TCN Block!"
            # print("x.shape, speaker embeds shape", x.shape, speaker_embeds.shape)
            speaker_embeds = speaker_embeds.repeat(1, 1, x.shape[-1])
            outputs = torch.cat([x, speaker_embeds], 1) # R^Kx(O+D)
        outputs = self.conv_in(outputs)
        outputs = self.prelu1(outputs)
        outputs = self.global_layer_norm1(outputs)
        outputs = self.de_cnn(outputs)
        outputs = self.prelu2(outputs)
        outputs = self.global_layer_norm2(outputs)
        outputs = self.conv_out(outputs)
        outputs = outputs + x
        return outputs

        

class SpeakerExtractor(nn.Module):
    def __init__(self, speech_enc_out_channels, in_channels, out_channels, speak_embed_dim, n_tcns=4, n_tcn_blocks=8) -> None:
        super(SpeakerExtractor, self).__init__()
        '''
        Speaker extractor is designed to estimate masks M_i for the target speaker in each scale i = 1, 2, 3, 
        conditioned on the embedding coefficients Y and the speaker embedding v. 
        Similar to Conv-TasNet [13], our speaker extractor repeats a stack of
        temporal convolutional network (TCN) blocks with a number of B for R times.
        In each stack, the TCN has a exponential growth dilation factor 2^b (b ∈ {0, · · · , B - 1}) 
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
        # in_channels = O, out_channels = P
        self.norm = LayerNorm(3 * speech_enc_out_channels)
        self.conv_in = Conv1d(3 * speech_enc_out_channels,
                              in_channels,
                              kernel_size=1) # 3 * N, O
        self.tcns = nn.ModuleList(
            TCNBlock(in_channels, out_channels, speak_embed_dim, is_first = not j) for i in range(n_tcn_blocks) for j in range(n_tcns)
        )
        # from tcn block we get an object with in_channels, so in_channels in masks is in_channels

        # As the output channels O from the last TCN block may be different from the channels N 
        # of the encoded representations Ei ∈ RK×N , 
        # we apply one 1 × 1 CNN to transform the dimension of the output channels 
        # from the last TCN block to be same as the encoded representations
        self.mask1 = Conv1d(in_channels, speech_enc_out_channels, kernel_size=1) # O, N, 1
        self.mask2 = Conv1d(in_channels, speech_enc_out_channels, kernel_size=1) # O, N, 1
        self.mask3 = Conv1d(in_channels, speech_enc_out_channels, kernel_size=1) # O, N, 1



    def forward(self, short, middle, long, speaker_embeds):
        # n x 1 x S => n x N x T
        w1 = F.relu(short)
        w2 = F.relu(middle)
        w3 = F.relu(long)
        # print(short.shape, middle.shape, long.shape)

        # apply layer norm by channels, input shape: [*, in_channels]
        speech = torch.cat([short, middle, long], dim=1)
        speech = speech.transpose(1, 2)
        speech = self.norm(speech)
        speech = speech.transpose(1, 2)
        speech = self.conv_in(speech)

    
        outputs = speech
        for layer in self.tcns:
            if layer.is_first:
                outputs = layer(outputs, speaker_embeds)
            else:
                outputs = layer(outputs)
        
        m1 = F.relu(self.mask1(speech))
        m2 = F.relu(self.mask2(speech))
        m3 = F.relu(self.mask3(speech))

        # point-wise multiplication
        s1, s2, s3 = w1 * m1, w2 * m2, w3 * m3
        return s1, s2, s3


# should be inherit sth from BaseModel?
class SpexPlus(nn.Module):
    def __init__(self, 
                 L1: int,
                 L2: int,
                 L3: int,
                 speech_enc_out_channels: int, # N
                 speak_ex_in_channels: int, # O
                 speak_ex_out_channels: int, # P,
                 speak_embed_dim: int, 
                 n_tcns: int, # B 
                 n_tcn_blocks: int, # R
                 num_speakers: int, 
                ) -> None:
        super(SpexPlus, self).__init__()
        self.speech_encoder = SpeechEncoder(L1, L2, L3, speech_enc_out_channels)
        self.speaker_encoder = SpeakerEncoder(speech_enc_out_channels, speak_ex_in_channels, speak_ex_out_channels, speak_embed_dim)
        self.speaker_extractor = SpeakerExtractor(speech_enc_out_channels=speech_enc_out_channels,
                                                  in_channels=speak_ex_in_channels,
                                                  out_channels=speak_ex_out_channels,
                                                  speak_embed_dim=speak_embed_dim,
                                                  n_tcns=n_tcns,
                                                  n_tcn_blocks=n_tcn_blocks)
        

        # batch_size x N x T -> batch_size x 1 x (T - 1) * L // 2 + L
        self.speech_decoder_short = Conv1d(speech_enc_out_channels, 1, kernel_size=L1, stride=L1 // 2, bias=True)
        self.speech_decoder_middle = Conv1d(speech_enc_out_channels, 1, kernel_size=L2, stride=L1 // 2, bias=True)
        self.speech_decoder_long = Conv1d(speech_enc_out_channels, 1, kernel_size=L3, stride=L1 // 2, bias=True)

        self.classifier = nn.Sequential(
            nn.Linear(speak_embed_dim, num_speakers),
            nn.Softmax(dim=1), # TODO: params
        )

    

    def forward(self, **batch) -> dict:
        reference = batch["ref_audio"]
        mixture = batch["mix_audio"]

        # print('initial shapes for ref and mix', reference.shape, mixture.shape)
        # x -- reference speech
        # y -- mixture speech
        x, y = self.speech_encoder(reference), self.speech_encoder(mixture)
        speaker_embeds = self.speaker_encoder(*x)
        short, middle, long = self.speaker_extractor(*y, speaker_embeds)
        speaker_embeds = speaker_embeds.squeeze(-1)
        logits = self.classifier(speaker_embeds)
        short, middle, long = self.speech_decoder_short(short), self.speech_decoder_middle(middle), self.speech_decoder_long(long)
        # print('final shapes', short.shape, middle.shape, long.shape)
        # batch_size, n_channels, len
        outputs = {
            "short": short, 
            "middle": middle,
            "long": long,
            "logits": logits
        }
        return outputs
