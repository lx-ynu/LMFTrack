import os

from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    data_root = os.environ.get('LMFTRACK_DATA_DIR', os.path.join(project_root, 'data'))
    output_root = os.environ.get('LMFTRACK_OUTPUT_DIR', os.path.join(project_root, 'output'))

    settings = EnvSettings()
    settings.prj_dir = project_root
    settings.save_dir = output_root
    settings.check_dir = os.path.join(project_root, 'model')
    settings.results_path = os.path.join(output_root, 'test', 'tracking_results')
    settings.segmentation_path = os.path.join(output_root, 'test', 'segmentation_results')
    settings.network_path = os.path.join(output_root, 'test', 'networks')
    settings.result_plot_path = os.path.join(output_root, 'test', 'result_plots')
    settings.lasher_path = os.path.join(data_root, 'LasHeR')
    settings.rgbt234_path = os.path.join(data_root, 'RGBT234')
    settings.gtot_path = os.path.join(data_root, 'GTOT')
    settings.vtuavst_path = os.path.join(data_root, 'VTUAV')
    return settings
