import logging
from itertools import combinations

import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

from poster_analyzer.recognizers.abstract import AbstractRecognizer, RecognizedComponent

logger = logging.getLogger(__name__)


class TextRecognizer(AbstractRecognizer):
    def __init__(self, resource_units: int = 1, context: str = ""):
        super().__init__(resource_units=resource_units, context=context)
        self.reader = easyocr.Reader(["en"])

    def enhance_image(self, image: Image.Image, contrast_factor=2.0) -> Image.Image:
        image = image.convert("RGBA")
        contrast = ImageEnhance.Contrast(image)
        return contrast.enhance(contrast_factor)

    def _ocr_image(self, image: Image.Image) -> list[tuple[tuple[float, float], str, float]]:
        return self.reader.readtext(np.array(image), detail=1, paragraph=False)

    def should_group_components(self, comp1: RecognizedComponent, comp2: RecognizedComponent) -> bool:
        return False

    def group_components(self, comp1: RecognizedComponent, comp2: RecognizedComponent) -> list[RecognizedComponent]:
        pass

    def guess_text_alignment(self, comp1: RecognizedComponent, comp2: RecognizedComponent) -> str:
        pass

    def derive_closest_snap():
        pass

    def recognize(self, image: Image.Image, min_probability=0.70) -> list[RecognizedComponent]:
        enhanced_image = self.enhance_image(image)
        ocr_results = self._ocr_image(enhanced_image)

        ungrouped_components = []
        for bbox, text, prob in ocr_results:
            if prob < min_probability:
                logger.debug("Skipping low probability text: %s (%.2f)", text, prob)
                continue

            x_min = min(point[0] for point in bbox)
            y_min = min(point[1] for point in bbox)
            x_max = max(point[0] for point in bbox)
            y_max = max(point[1] for point in bbox)
            coordinates = (x_min, y_min, x_max, y_max)

            logger.debug("Recognized text: %s (%.2f) at %s", text, prob, coordinates)
            ungrouped_components.append(
                RecognizedComponent(
                    component_type="text",
                    coordinates=coordinates,
                    probability=prob,
                    additional_data={
                        "text": text,
                        "text_alignment": "left",
                    },
                ),
            )

        logger.debug("Found total ungrouped text components: %d", len(ungrouped_components))

        for comp1, comp2 in combinations(ungrouped_components, 2):
            if self.should_group_components(comp1, comp2):
                grouped = self.group_components(comp1, comp2)
                ungrouped_components.remove(comp1)
                ungrouped_components.remove(comp2)
                ungrouped_components.extend(grouped)

        grouped_components = ungrouped_components

        logger.debug("Total text components after grouping: %d", len(grouped_components))

        return grouped_components

    def debug_draw_bounding_boxes(self, image: Image.Image, components: list[RecognizedComponent]) -> Image.Image:
        logger.debug("Drawing %d bounding boxes for debug", len(components))

        enhanced_image = self.enhance_image(image)
        draw = ImageDraw.Draw(enhanced_image)
        for comp in components:
            text = comp.additional_data.get("text", "")
            logger.debug("Text: %s, Probability: %s, BBox: %s", text, comp.probability, comp.coordinates)
            draw.rectangle(comp.coordinates, outline="red", width=2)
        return enhanced_image
