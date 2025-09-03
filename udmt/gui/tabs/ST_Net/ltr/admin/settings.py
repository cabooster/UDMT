from udmt.gui.tabs.ST_Net.ltr.admin.environment import env_settings
# from ltr.admin.environment import env_settings


class Settings:
    """ Training settings, e.g. the paths to datasets and networks."""
    def __init__(self,run_training_params):
        self.set_default(run_training_params)

    def set_default(self,run_training_params):
        self.env = env_settings(run_training_params)
        self.use_gpu = True


