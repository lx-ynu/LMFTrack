import os
import re
from pathlib import Path

import numpy as np

from lib.test.evaluation.data import BaseDataset, Sequence, SequenceList
from lib.test.utils.load_text import load_text


def _natural_key(path):
    name = Path(path).name
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', name)]


class LasHeRDataset(BaseDataset):
    """LasHeR test-set loader for RGB-T tracking."""

    def __init__(self, split='testingset'):
        super().__init__()
        if split not in {'testingset', 'val'}:
            raise ValueError("The released evaluator supports the LasHeR test split only.")
        self.split = 'testingset' if split == 'val' else split
        self.dataset_root = Path(self.env_settings.lasher_path)
        self.base_path = self.dataset_root / self.split
        self.sequence_list = self._get_sequence_list()

    def _get_sequence_list(self):
        list_file = self.dataset_root / f'{self.split}List.txt'
        if not list_file.is_file():
            raise FileNotFoundError(f'LasHeR sequence list not found: {list_file}')
        return [line.strip() for line in list_file.read_text(encoding='utf-8').splitlines() if line.strip()]

    def _construct_sequence(self, sequence_name):
        sequence_path = self.base_path / sequence_name
        annotation = sequence_path / 'init.txt'
        visible_dir = sequence_path / 'visible'
        infrared_dir = sequence_path / 'infrared'
        for path in (annotation, visible_dir, infrared_dir):
            if not path.exists():
                raise FileNotFoundError(f'LasHeR sequence resource not found: {path}')

        ground_truth = load_text(str(annotation), delimiter=',', dtype=np.float64).reshape(-1, 4)
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        visible_frames = sorted(
            [str(path) for path in visible_dir.iterdir() if path.suffix.lower() in extensions], key=_natural_key
        )
        infrared_frames = sorted(
            [str(path) for path in infrared_dir.iterdir() if path.suffix.lower() in extensions], key=_natural_key
        )
        if len(visible_frames) != len(infrared_frames):
            raise RuntimeError(f'RGB/TIR frame-count mismatch in LasHeR sequence: {sequence_name}')
        return Sequence(sequence_name, [visible_frames, infrared_frames], 'lasher', ground_truth)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(name) for name in self.sequence_list])

    def __len__(self):
        return len(self.sequence_list)
