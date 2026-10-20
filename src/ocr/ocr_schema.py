import math
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator

MIN_BBOX_DIMENSION = 25
Coordinate = Annotated[float, Field(ge=0, le=1000)]


class Sketch(BaseModel):
    index: int = Field(ge=1)
    box_2d: list[Coordinate] = Field(
        min_length=4,
        max_length=4,
        description="Normalized [y_min, x_min, y_max, x_max] coordinates from 0 to 1000.",
    )

    @field_validator("box_2d")
    @classmethod
    def validate_finite_box_coordinates(cls, box: list[float]) -> list[float]:
        if not all(math.isfinite(value) for value in box):
            raise ValueError("Bounding-box coordinates must be finite")
        return box

    @model_validator(mode="after")
    def validate_box_dimensions(self) -> "Sketch":
        y_min, x_min, y_max, x_max = self.box_2d
        width = x_max - x_min
        height = y_max - y_min
        if width <= 0 or height <= 0:
            raise ValueError("Bounding-box coordinates must have positive dimensions")
        if width < MIN_BBOX_DIMENSION or height < MIN_BBOX_DIMENSION:
            raise ValueError(
                f"Bounding boxes must be at least {MIN_BBOX_DIMENSION} units wide and high"
            )
        return self


class OCRPageResult(BaseModel):
    markdown_content: str
    sketches: list[Sketch] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_sketch_indices(self) -> "OCRPageResult":
        indices = [sketch.index for sketch in self.sketches]
        if len(indices) != len(set(indices)):
            raise ValueError("Sketch indices must be unique")
        return self
