import yaml

from customlogger.logger import Logger

logger = Logger(__name__)
class Config:
    """
        A class to hold configuration parameters loaded from a YAML file.
    """
    def __init__(self, config_path):
        config_dict = self._read_config(config_path)
        for key, value in config_dict.items():
            setattr(self, key, value)
        logger.info(f"Configuration loaded successfully from {config_path}")
    def _read_config(self, config_path):
        with open(config_path, "r") as config_file:
            config_dict = yaml.safe_load(config_file)
        return config_dict

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__})'