import importlib


def env_settings():
    """Load the portable training paths defined in lib/train/admin/local.py."""
    return importlib.import_module('lib.train.admin.local').EnvironmentSettings()
