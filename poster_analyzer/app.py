import logging
import os

from PIL import Image

from .recognizers import RecognizedComponent, TextRecognizer


def setup_debug_logging():
    logger = logging.getLogger("poster_analyzer")
    logger.setLevel(os.getenv("POSTER_ANALYZER_LOG_LEVEL", "DEBUG"))
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    return logger

IS_DEBUG = os.getenv("POSTER_ANALYZER_DEBUG", "0").lower() in ["1", "true", "yes"]
logger = setup_debug_logging() if IS_DEBUG else logging.getLogger("poster_analyzer")


def recognize_all_components(source: str | Image.Image, context: str = "") -> list[RecognizedComponent]:
    image = Image.open(source) if isinstance(source, str) else source
    components = []

    tr = TextRecognizer(context=context)
    components.extend(tr.recognize(image))

    if IS_DEBUG:
        logger.debug("Recognized %d text components", len(components))
        debug_image = tr.debug_draw_bounding_boxes(image.copy(), components)
        debug_image.show()


    return components
