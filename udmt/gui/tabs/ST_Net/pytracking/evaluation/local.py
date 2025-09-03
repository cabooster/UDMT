# from pytracking.evaluation.environment import EnvSettings
from udmt.gui.tabs.ST_Net.pytracking.evaluation.environment import EnvSettings
from udmt.gui import BASE_DIR
def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = '' #'D:/tracking_datasets/Tracking/GOT-10k-test' #!!
    settings.got_packed_results_path = '../results'
    settings.got_reports_path = '../reports'
    settings.lasot_path = ''
    settings.network_path = ''#BASE_DIR+ '/networks' #!!
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.result_plot_path = '../results'
    settings.results_path = '../results'
    settings.segmentation_path = ''
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings

