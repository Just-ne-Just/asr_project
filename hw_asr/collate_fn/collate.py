import logging
from typing import List
from torch.nn.utils.rnn import pad_sequence
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    audio = []
    spec = []
    text_encoded = []
    text_encoded_length = []
    text = []

    for item in dataset_items:
        audio.append(item['audio'][0])
        spec.append(item['spectrogram'][0].T)
        text_encoded.append(item['text_encoded'][0])
        text_encoded_length.append(len(item['text_encoded'][0]))
        text.append(item['text'])
    
    audio = pad_sequence(audio, batch_first=True)
    spec = pad_sequence(spec, batch_first=True).transpose(1, 2)
    text_encoded = pad_sequence(text_encoded, batch_first=True)
        
    return {
        "audio": audio,
        "spectrogram": spec,
        "text_encoded": text_encoded,
        "text_encoded_length": torch.tensor(text_encoded_length),
        "text": text
    }
