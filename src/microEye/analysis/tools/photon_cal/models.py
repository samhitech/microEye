from dataclasses import asdict, dataclass


@dataclass
class CalibrationDatasetMeta:
    name: str
    signal_path: str
    dark_path: str
    laser_power_mW: float | None = None
    port_power_mW: float | None = None
    pixel_size_um: float | None = None
    wavelength_nm: float = 488.0
    min_exposure_s: float | None = None

    dark_calibration_directory: str | None = None
    gain: float | None = None
    responsivity: float | None = None
    quantum_efficiency: float | None = None

    def to_payload(self) -> dict:
        return asdict(self)
