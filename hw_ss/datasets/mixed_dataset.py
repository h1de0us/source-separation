import logging
from pathlib import Path
import os

import torch
import torchaudio

from torch.utils.data import Dataset
from hw_ss.utils.parse_config import ConfigParser
from hw_ss.utils import ROOT_PATH

logger = logging.getLogger(__name__)

class LibrispeechMixDataset(Dataset):
    def __init__(self, 
                 path,
                 config_parser: ConfigParser,
                 wave_augs=None,
                 spec_augs=None,) -> None:
        super().__init__()
        self.path = os.path.join(ROOT_PATH, path)
        print(self.path)
        files = os.listdir(os.path.join(ROOT_PATH, path))
        files = ['-'.join(file.split('-')[:-1]) for file in files]
        files = list(set(files))
        self._index = files
        self.config_parser = config_parser
        self.wave_augs = wave_augs
        self.spec_augs = spec_augs
        speakers = list(set([file.split('_')[0] for file in files]))
        speakers = {speaker: cls for cls, speaker in enumerate(speakers)}
        self.speakers_ids = speakers


    def __getitem__(self, ind) -> dict:
        assert ind < len(self)
        path_prefix = self._index[ind]
        speaker_prefix = path_prefix.split('_')[0]
        librispeech_speaker_index = self.speakers_ids[speaker_prefix]
        print(f'index for speaker {speaker_prefix} is {librispeech_speaker_index}')
        mix_path = os.path.join(self.path, path_prefix + "-mixed.wav")
        target_path = os.path.join(self.path, path_prefix + "-target.wav")
        ref_path = os.path.join(self.path, path_prefix + "-ref.wav")
        mix_wave, target_wave, ref_wave = self.load_audio(mix_path), self.load_audio(target_path), self.load_audio(ref_path)
        return {
            "path_prefix": path_prefix,
            "mix_audio": mix_wave,
            "target_audio": target_wave,
            "ref_audio": ref_wave,
            "librispeech_speaker_index": librispeech_speaker_index
        }
    

    def __len__(self):
        return len(self._index)
    

    def load_audio(self, path) -> torch.Tensor:
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor
