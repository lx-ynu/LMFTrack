import re
from pathlib import Path

import numpy as np

from lib.test.evaluation.data import BaseDataset, Sequence, SequenceList
from lib.test.utils.load_text import load_text


def _natural_key(path):
    name = Path(path).name
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', name)]


def _first_existing(base, names):
    for name in names:
        candidate = base / name
        if candidate.is_dir():
            return candidate
    return None


class _VTUAVDataset(BaseDataset):
    """VTUAV loader with common-layout auto-detection."""

    def __init__(self, subset):
        super().__init__()
        self.subset = subset
        self.base_path = self._resolve_sequence_root(Path(self.env_settings.vtuavst_path))
        self.sequence_list = sorted(
            [path.name for path in self.base_path.iterdir() if path.is_dir() and self._is_sequence(path)],
            key=_natural_key,
        )
        if not self.sequence_list:
            raise RuntimeError(
                f'No VTUAV {self.subset} sequences were found under {self.base_path}. '
                'Please update vtuavst_path in lib/test/evaluation/local.py.'
            )

    @staticmethod
    def _is_sequence(path):
        visible = _first_existing(path, ('visible', 'rgb', 'v'))
        infrared = _first_existing(path, ('infrared', 'ir', 'thermal', 'i'))
        return visible is not None and infrared is not None

    def _resolve_sequence_root(self, base):
        if self.subset == 'short-term':
            names = ('short-term', 'short_term', 'shortterm', 'short')
        else:
            names = ('long-term', 'long_term', 'longterm', 'long')

        candidates = []
        for name in names:
            candidates.extend((base / 'test' / name, base / name / 'test', base / name))
        candidates.extend((base / 'test', base))

        for candidate in candidates:
            if candidate.is_dir() and any(path.is_dir() and self._is_sequence(path) for path in candidate.iterdir()):
                return candidate
        raise FileNotFoundError(f'VTUAV {self.subset} directory not found below: {base}')

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(name) for name in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = self.base_path / sequence_name
        visible_dir = _first_existing(sequence_path, ('visible', 'rgb', 'v'))
        infrared_dir = _first_existing(sequence_path, ('infrared', 'ir', 'thermal', 'i'))

        annotation = None
        for name in ('init.txt', 'groundtruth.txt', 'groundtruth_rect.txt', 'groundtruth_rect.1.txt'):
            candidate = sequence_path / name
            if candidate.is_file():
                annotation = candidate
                break
        if annotation is None:
            raise FileNotFoundError(f'Annotation file not found for VTUAV sequence: {sequence_name}')

        first_line = annotation.read_text(errors='ignore').splitlines()[0]
        delimiter = ',' if ',' in first_line else None
        ground_truth = load_text(str(annotation), delimiter=delimiter, dtype=np.float64)
        extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        visible_frames = sorted(
            [str(path) for path in visible_dir.iterdir() if path.suffix.lower() in extensions], key=_natural_key
        )
        infrared_frames = sorted(
            [str(path) for path in infrared_dir.iterdir() if path.suffix.lower() in extensions], key=_natural_key
        )
        if len(visible_frames) != len(infrared_frames):
            raise RuntimeError(f'RGB/TIR frame-count mismatch in VTUAV sequence: {sequence_name}')

        dataset_name = 'vtuavst' if self.subset == 'short-term' else 'vtuavlt'
        return Sequence(sequence_name, [visible_frames, infrared_frames], dataset_name, ground_truth.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)


class VTUAVSTDataset(_VTUAVDataset):
    def __init__(self):
        super().__init__('short-term')


class VTUAVLTDataset(_VTUAVDataset):
    def __init__(self):
        super().__init__('long-term')
