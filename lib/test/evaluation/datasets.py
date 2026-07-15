from collections import namedtuple
import importlib

from lib.test.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])
MODULE_PATTERN = 'lib.test.evaluation.%sdataset'

# Benchmarks reported in the LMFTrack paper.
dataset_dict = {
    'lasher': DatasetInfo(MODULE_PATTERN % 'lasher', 'LasHeRDataset', {'split': 'testingset'}),
    'lasher_test': DatasetInfo(MODULE_PATTERN % 'lasher', 'LasHeRDataset', {'split': 'testingset'}),
    'rgbt234': DatasetInfo(MODULE_PATTERN % 'rgbt234', 'RGBT234Dataset', {}),
    'gtot': DatasetInfo(MODULE_PATTERN % 'gtot', 'GTOTDataset', {}),
    'vtuavst': DatasetInfo(MODULE_PATTERN % 'vtuavst', 'VTUAVSTDataset', {}),
    'vtuavlt': DatasetInfo(MODULE_PATTERN % 'vtuavst', 'VTUAVLTDataset', {}),
}


def load_dataset(name: str):
    name = name.lower()
    dataset_info = dataset_dict.get(name)
    if dataset_info is None:
        supported = ', '.join(sorted(dataset_dict))
        raise ValueError(f"Unknown dataset '{name}'. Supported datasets: {supported}")

    module = importlib.import_module(dataset_info.module)
    dataset = getattr(module, dataset_info.class_name)(**dataset_info.kwargs)
    return dataset.get_sequence_list()


def get_dataset(*names):
    dataset = SequenceList()
    for name in names:
        dataset.extend(load_dataset(name))
    return dataset
