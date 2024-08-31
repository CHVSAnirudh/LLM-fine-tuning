from transformers import (
    AutoTokenizer
)
from lib.logger import logger

model = "NousResearch/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
    model,
    use_fast=True,
    trust_remote=True
)
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True


logger.info(tokenizer)

logger.info(
    tokenizer.apply_chat_template([
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris"}
    ],
    tokenize=False
))
