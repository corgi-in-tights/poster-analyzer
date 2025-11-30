import logging
import os

from PIL import Image

from .text import get_text_bounding_boxes


def setup_debug_logging():
    logger = logging.getLogger("poster_generator")
    logger.setLevel(os.getenv("POSTER_GENERATOR_LOG_LEVEL", "DEBUG"))
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    return logger


IS_DEBUG = os.getenv("POSTER_ANALYZER_DEBUG", "0").lower() in ["1", "true", "yes"]

logger = setup_debug_logging() if IS_DEBUG else logging.getLogger("poster_analyzer")


def analyze_image(path):
    image = Image.open(path)
    return get_text_bounding_boxes(image)
