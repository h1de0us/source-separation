import logging
from typing import List
import torch

logger = logging.getLogger(__name__)

def pad_sequence(batch):
    batch = [item.T for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.transpose(1, 2)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    Dataset entry:
    "path_prefix",
    "mix_audio",
    "target_audio",
    "ref_audio",
    "librispeech_speaker_index"
    """


    result_batch = {
        'mix_audio': pad_sequence([item['mix_audio'] for item in dataset_items]),
        'target_audio': pad_sequence([item['target_audio'] for item in dataset_items]),
        'ref_audio': pad_sequence([item['target_audio'] for item in dataset_items]),
        # 'duration': [item['duration'] for item in dataset_items],
        'librispeech_speaker_index': torch.as_tensor([item["librispeech_speaker_index"] for item in dataset_items]),
        'audio_path': [item['path_prefix'] for item in dataset_items] # idk, for debugging purposes
    }
    return result_batch