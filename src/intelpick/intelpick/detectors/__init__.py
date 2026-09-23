from dataclasses import dataclass


@dataclass
class Blob:
    label: str
    confidence: float
    u: float       # centroid, pixels
    v: float
    angle: float   # radians
    area: float    # pixels
