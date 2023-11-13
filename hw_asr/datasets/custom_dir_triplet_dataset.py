import logging
from pathlib import Path

from hw_asr.datasets.custom_audio_dataset import CustomAudioDataset
import os

logger = logging.getLogger(__name__)


class CustomDirTripletDataset(CustomAudioDataset):
    def __init__(self, mix, ref, target, *args, **kwargs):
        data = []

        mix_paths = sorted(os.listdir(mix))
        ref_paths = sorted(os.listdir(ref))
        target_paths = sorted(os.listdir(target))

        for mix_path, ref_path, target_path in zip(mix_paths, ref_paths, target_paths):
            data.append({
                "mix": f"{mix}/{mix_path}",
                "reference": f"{ref}/{ref_path}",
                "target": f"{target}/{target_path}",
                "speaker_id": -1
            })
        
        super().__init__(data, *args, **kwargs)
