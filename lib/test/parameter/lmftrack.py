import os

from lib.config.lmftrack.config import cfg, update_config_from_file
from lib.test.evaluation.environment import env_settings
from lib.test.utils import TrackerParams


def parameters(yaml_name: str):
    params = TrackerParams()
    environment = env_settings()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    yaml_file = os.path.join(project_root, 'experiments', 'lmftrack', f'{yaml_name}.yaml')
    update_config_from_file(yaml_file)
    params.cfg = cfg

    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.template_update_threshold = cfg.TEST.TEMPLATE_UPDATE_THRESHOLD

    checkpoint_name = f'LMFTrack_ep{cfg.TEST.EPOCH:04d}.pth.tar'
    params.checkpoint = os.path.join(
        environment.save_dir,
        yaml_name,
        'checkpoints',
        'train',
        'lmftrack',
        yaml_name,
        checkpoint_name,
    )
    params.save_all_boxes = False
    return params
