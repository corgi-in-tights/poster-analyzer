from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL.Image import Image


@dataclass
class RecognizedComponent:
    component_type: str
    coordinates: tuple[float, float, float, float]
    probability: float
    additional_data: dict | None = None


class AbstractRecognizer(ABC):
    def __init__(self, resource_units: int = 1, context: str = ""):
        self.resource_units = resource_units
        self.context = context

    @abstractmethod
    def recognize(self, image: Image, min_probability=0.85) -> list[RecognizedComponent]:
        ...
