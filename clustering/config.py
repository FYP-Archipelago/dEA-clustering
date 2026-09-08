from __future__ import annotations

import yaml
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field

Auto = Literal["auto"]


class LSHConfig(BaseModel):
    enabled: bool = False
    hashes: int = Field(default=6, ge=1)
    radius: float | Auto = "auto"
    width_percentile: float = Field(default=20.0, gt=0, lt=100)
    hamming_bit_multiplier: int = Field(default=4, ge=1)
    num_perm: int = Field(default=128, ge=8)


class BirchConfig(BaseModel):
    enabled: bool = False
    threshold: float | Auto = "auto"
    branching_factor: int = Field(default=50, ge=2)


class DenStreamConfig(BaseModel):
    enabled: bool = False
    epsilon: float | Auto = "auto"
    mu: float = Field(default=10.0, gt=0)
    beta: float = Field(default=0.5, gt=0, le=1)
    half_life: float | Auto = "auto"
    clock: Literal["arrival", "t_wall", "generation"] = "t_wall"


class LayoutConfig(BaseModel):
    method: Literal["none", "pca", "mds"] = "pca"
    mds_landmarks: int = Field(default=1500, ge=10)


class CalibrationConfig(BaseModel):
    sample_size: int = Field(default=2000, ge=50)
    percentile: float = Field(default=50.0, gt=0, lt=100)


class Config(BaseModel):
    lsh: LSHConfig = LSHConfig()
    birch: BirchConfig = BirchConfig()
    denstream: DenStreamConfig = DenStreamConfig()
    layout: LayoutConfig = LayoutConfig()
    calibration: CalibrationConfig = CalibrationConfig()

    seed: int = 0

    @property
    def any_stage_enabled(self) -> bool:
        return self.lsh.enabled or self.birch.enabled or self.denstream.enabled

    def stage_flags(self) -> dict[str, bool]:
        return {
            "lsh": self.lsh.enabled,
            "birch": self.birch.enabled,
            "denstream": self.denstream.enabled,
        }

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        """Loads the config from the .yaml file passed with -c flag"""
        if path is None:
            return cls()
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(data)
