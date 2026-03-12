from loguru import logger
import sys

def setup_logger():
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    return logger
