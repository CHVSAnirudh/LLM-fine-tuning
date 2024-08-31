import torch

from customlogger.logger import (Logger)

logger = Logger(__name__)
def clear_gpu_cache(arr: list, clear_cache: bool = True):
    """
    Use this function when you need to delete the objects, free their memory
    and also delete the cuda cache
    """
    for obj in arr:
        logger.info(f"Deleting {obj}")
        del obj
    if clear_cache:
        torch.cuda.empty_cache()
        logger.info("Cleared Cuda Cache")


def is_gpu_bf16_supported() -> bool:
    """
    Check if the GPU supports BF16
    """
    if getattr(torch, "float16") == torch.float16:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            logger.info("Your GPU supports bfloat16")
            return True
        else:
            logger.info("Your GPU does not support bfloat16")
            return False