import logging
import os


def get_logger(name: str = "ai-server") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(level)
    return logger
