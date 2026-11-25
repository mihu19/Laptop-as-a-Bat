"""Validated near-field settings shared by the live app and diagnostic."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("sonar_config.json")


@dataclass(frozen=True)
class SonarSettings:
    gain: float = 10**(-48/20)
    alpha: float = .02
    low_hz: int = 17500
    high_hz: int = 21500
    noise_multiplier: float = 4.
    blanking_m: float = .15
    persistence: float = .64
    smoothing: float = 0.

    def __post_init__(self):
        if not (0 < self.gain <= 10**(-30/20)
                and 0.004 <= self.alpha <= .05
                and 12000 <= self.low_hz < self.high_hz <= 22000
                and self.high_hz-self.low_hz >= 3000
                and 3 <= self.noise_multiplier <= 6
                and .15 <= self.blanking_m <= .18
                and .5 <= self.persistence <= .86
                and 0 <= self.smoothing <= 1.4):
            raise ValueError("Sonar settings outside supported bounds.")

    @property
    def low_band(self):
        return self.low_hz < 17000

    @property
    def carriers(self):
        if self.ml_band:
            return (13000., 15500.)
        return (14800., 16800.) if self.low_band else (18500., 20500.)

    @property
    def ml_band(self):
        return self.low_hz < 14000


LOW_BAND = SonarSettings(low_hz=14500, high_hz=17500)
ML_BAND = SonarSettings(low_hz=12000, high_hz=16500)


def load_settings(path=CONFIG_PATH):
    try:
        return SonarSettings(**json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return ML_BAND
    except (ValueError, TypeError) as error:
        raise ValueError(f"Invalid sonar configuration in {path}: {error}") from error


def save_settings(settings, path=CONFIG_PATH):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(asdict(settings), indent=2)+"\n", encoding="utf-8")
    pending.replace(path)
