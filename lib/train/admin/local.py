import os


class EnvironmentSettings:
    def __init__(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
        data_root = os.environ.get('LMFTRACK_DATA_DIR', os.path.join(project_root, 'data'))
        output_root = os.environ.get('LMFTRACK_OUTPUT_DIR', os.path.join(project_root, 'output'))

        self.workspace_dir = output_root
        self.tensorboard_dir = os.path.join(output_root, 'tensorboard')
        self.pretrained_networks = os.path.join(project_root, 'pretrained_models')
        self.lasher_train_dir = os.path.join(data_root, 'LasHeR', 'trainingset')
        self.lasher_test_dir = os.path.join(data_root, 'LasHeR', 'testingset')
