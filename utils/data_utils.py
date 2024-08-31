import os
from dotenv import load_dotenv
from datasets import load_dataset

from customlogger.logger import Logger

logger = Logger(__name__)

def format_example(example, tokenizer):
    chat_resp = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["subject_line"]}
        ],
        tokenize=False
    )
    assert isinstance(chat_resp, str)
    return {
        "text": chat_resp,
    }


class DatasetLoader:
    def __init__(self, dataset_path:str, use_hf:bool, tokenizer):
        load_dotenv()
        self.dataset_path = dataset_path
        self.use_hf = use_hf
        self.tokenizer = tokenizer
        self.dataset_loaded = False
        self._load_dataset()

    def _load_dataset(self):
        # Load training split (you can process it here)
        self.train_dataset = load_dataset(
            self.dataset_path, use_auth_token=os.getenv("HF_AUTH_TOKEN"))
        self.train_dataset = self.train_dataset.map(
            lambda example: format_example(example, self.tokenizer)
        )
        # drop columns
        self.train_dataset = self.train_dataset.remove_columns(
            ['prompt', 'response']
        )
        logger.debug(self.train_dataset)
        logger.debug(self.train_dataset["train"][0])
        # self.data = self.train_dataset.train_test_split(test_size=0.2)
        self.data = {
            "train": self.train_dataset["train"],
            "test": self.train_dataset["test"]
        }
        # Set the dataset flag
        self.dataset_loaded = True

    def get_dataset(self):
        assert self.dataset_loaded, \
            "Dataset not loaded. Please run load_dataset() first."
        return self.data['train'], self.data['test']
        # return self.data, self.data


