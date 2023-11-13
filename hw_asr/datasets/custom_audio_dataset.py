import logging
from pathlib import Path

import torchaudio

from hw_asr.base.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CustomAudioDataset(BaseDataset):
    def __init__(self, data, *args, **kwargs):
        index = data
        for entry in data:
            entry["mix"] = str(Path(entry["mix"]).absolute().resolve())
            entry["reference"] = str(Path(entry["reference"]).absolute().resolve())
            entry["target"] = str(Path(entry["target"]).absolute().resolve())

        super().__init__(index, *args, **kwargs)
