import importlib
import os


class EnvSettings:
    """Container for evaluation paths."""

    def __init__(self):
        test_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.prj_dir = ''
        self.save_dir = ''
        self.check_dir = ''
        self.results_path = os.path.join(test_root, 'tracking_results')
        self.segmentation_path = os.path.join(test_root, 'segmentation_results')
        self.network_path = os.path.join(test_root, 'networks')
        self.result_plot_path = os.path.join(test_root, 'result_plots')
        self.lasher_path = ''
        self.rgbt234_path = ''
        self.gtot_path = ''
        self.vtuavst_path = ''


def env_settings():
    """Load the portable evaluation paths defined in lib/test/evaluation/local.py."""
    return importlib.import_module('lib.test.evaluation.local').local_env_settings()
