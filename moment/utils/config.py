from yaml import CLoader as Loader
from yaml import dump, load


class Config:
    def __init__(
        self,
        config_file_path="configs/config.yaml",
        default_config_file_path="configs/default.yaml",
        verbose: bool = False,
    ):
        """
        Class to read and parse the config.yml file
        """
        self.config_file_path = config_file_path
        self.default_config_file_path = default_config_file_path
        self.verbose = verbose

    def parse(self):
        with open(self.config_file_path, "rb") as f:
            self.config = load(f, Loader=Loader)

        with open(self.default_config_file_path, "rb") as f:
            default_config = load(f, Loader=Loader)

        for key in default_config.keys():
            if self.config.get(key) is None:
                self.config[key] = default_config[key]
                if self.verbose:
                    print(f"Using default config for {key} : {default_config[key]}")

        return self.config

    def save_config(self):
        with open(self.config_file_path, "w") as f:
            dump(self.config, f)
