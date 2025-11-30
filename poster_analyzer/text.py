import logging

import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

logging = logging.getLogger(__name__)


def get_text_bounding_boxes(image: Image.Image):
    image = enhance_image(image, factor=3.0)

    result = ocr_image(image)

    image = draw_bounding_boxes(image, result)
    image.show()

    return result


def enhance_image(image, factor):
    image = image.convert("RGB")

    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def ocr_image(image):
    reader = easyocr.Reader(["en"])
    return reader.readtext(np.array(image), detail=1, paragraph=False)


def draw_bounding_boxes(image, results):
    draw = ImageDraw.Draw(image)
    for bbox, text, prob in results:
        logging.debug("Text: %s, Probability: %s, BBox: %s", text, prob, bbox)
        draw.rectangle([bbox[0], bbox[2]], outline="red", width=2)
    return image
