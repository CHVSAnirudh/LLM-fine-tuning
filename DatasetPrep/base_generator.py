class BaseDatasetGenerator:
    def __init__(self, config):
        self.config = config
        self.dataset = None

    def generate(self):
        raise NotImplementedError

    def get_dataset(self):
        return self.dataset