import re
from pathlib import Path

import numpy as np

from lib.test.evaluation.data import BaseDataset, Sequence, SequenceList
from lib.test.utils.load_text import load_text


def _natural_key(path):
    name = Path(path).name
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', name)]


class GTOTDataset(BaseDataset):
    """GTOT benchmark loader."""

    def __init__(self):
        super().__init__()
        self.base_path = Path(self.env_settings.gtot_path)
        if not self.base_path.is_dir():
            raise FileNotFoundError(f'GTOT directory not found: {self.base_path}')
        self.sequence_list = sorted([p.name for p in self.base_path.iterdir() if p.is_dir()], key=_natural_key)

    def _construct_sequence(self, sequence_name):
        sequence_path = self.base_path / sequence_name
        annotation = sequence_path / 'init.txt'
        visible_dir = sequence_path / 'v'
        infrared_dir = sequence_path / 'i'
        first_line = annotation.read_text(errors='ignore').splitlines()[0]
        delimiter = ',' if ',' in first_line else '\t'
        ground_truth = load_text(str(annotation), delimiter=delimiter, dtype=np.float64).reshape(-1, 4)
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        visible_frames = sorted([str(p) for p in visible_dir.iterdir() if p.suffix.lower() in extensions], key=_natural_key)
        infrared_frames = sorted([str(p) for p in infrared_dir.iterdir() if p.suffix.lower() in extensions], key=_natural_key)
        if len(visible_frames) != len(infrared_frames):
            raise RuntimeError(f'RGB/TIR frame-count mismatch in GTOT sequence: {sequence_name}')
        return Sequence(sequence_name, [visible_frames, infrared_frames], 'gtot', ground_truth)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(name) for name in self.sequence_list])

    def __len__(self):
        return len(self.sequence_list)
