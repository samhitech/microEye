import json
from datetime import datetime, timezone

import numpy as np

from microEye import __version__
from microEye.analysis.tools.photon_cal.constants import RESULTS_CACHE_SCHEMA_VERSION
from microEye.analysis.tools.photon_cal.models import CalibrationDatasetMeta


def _normalize_variance_source(value: str | None) -> str:
    source = str(value or 'shot').lower()
    return 'total' if source == 'total' else 'shot'


def dataset_from_payload(payload: dict) -> CalibrationDatasetMeta:
    return CalibrationDatasetMeta(
        name=str(payload.get('name', 'dataset')),
        signal_path=str(payload.get('signal_path', '')),
        dark_path=str(payload.get('dark_path', '')),
        laser_power_mW=(
            None
            if payload.get('laser_power_mW') is None
            else float(payload.get('laser_power_mW'))
        ),
        port_power_mW=(
            None
            if payload.get('port_power_mW') is None
            else float(payload.get('port_power_mW'))
        ),
        pixel_size_um=(
            None
            if payload.get('pixel_size_um') is None
            else float(payload.get('pixel_size_um'))
        ),
        wavelength_nm=float(payload.get('wavelength_nm', 488.0)),
        min_exposure_s=(
            None
            if payload.get('min_exposure_s') is None
            else float(payload.get('min_exposure_s'))
        ),
    )


def save_analysis_cache(
    file_path: str,
    datasets: dict[str, CalibrationDatasetMeta],
    returned_data: dict,
    exposure_times_s: np.ndarray,
    gain_variance_source: str | None = None,
) -> int:
    arrays: dict[str, np.ndarray] = {
        'exposure_times_s': np.asarray(exposure_times_s),
    }

    dataset_order = list(datasets.keys())
    dataset_prefixes = {}
    dataset_payloads = {}
    metric_keys = {}

    for index, name in enumerate(dataset_order):
        prefix = f'd{index:04d}'
        dataset_prefixes[name] = prefix
        dataset_payloads[name] = datasets[name].to_payload()

        params = returned_data.get(name, {})
        keys_for_dataset = []
        for key, value in params.items():
            arrays[f'{prefix}__{key}'] = np.asarray(value)
            keys_for_dataset.append(key)
        metric_keys[name] = keys_for_dataset

    if gain_variance_source is None:
        for params in returned_data.values():
            if (
                isinstance(params, dict)
                and params.get('fit_variance_source') is not None
            ):
                gain_variance_source = str(params.get('fit_variance_source'))
                break

    resolved_gain_variance_source = _normalize_variance_source(gain_variance_source)

    manifest = {
        'schema_version': RESULTS_CACHE_SCHEMA_VERSION,
        'tool': 'PhotonTransfer',
        'microeye_version': __version__,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_order': dataset_order,
        'dataset_prefixes': dataset_prefixes,
        'dataset_payloads': dataset_payloads,
        'metric_keys': metric_keys,
        'analysis_options': {
            'gain_variance_source': resolved_gain_variance_source,
        },
    }

    arrays['manifest'] = np.array(json.dumps(manifest, indent=2))
    np.savez(file_path, **arrays)
    return len(dataset_order)


def load_analysis_cache(
    file_path: str,
) -> tuple[dict[str, CalibrationDatasetMeta], dict, np.ndarray, dict]:
    with np.load(file_path, allow_pickle=False) as payload:
        if 'manifest' not in payload:
            raise ValueError('Missing manifest in NPZ file.')
        if 'exposure_times_s' not in payload:
            raise ValueError('Missing exposure_times_s in NPZ file.')

        manifest = json.loads(str(np.array(payload['manifest']).item()))
        schema_version = int(manifest.get('schema_version', 0))
        if schema_version > RESULTS_CACHE_SCHEMA_VERSION:
            raise ValueError(
                f'Unsupported schema version {schema_version}. '
                f'Max supported is {RESULTS_CACHE_SCHEMA_VERSION}.'
            )

        dataset_order: list[str] = manifest.get('dataset_order', [])
        dataset_prefixes: dict = manifest.get('dataset_prefixes', {})
        dataset_payloads: dict = manifest.get('dataset_payloads', {})
        metric_keys: dict = manifest.get('metric_keys', {})
        analysis_options: dict = manifest.get('analysis_options', {}) or {}

        resolved_gain_variance_source = _normalize_variance_source(
            analysis_options.get('gain_variance_source')
        )
        analysis_options['gain_variance_source'] = resolved_gain_variance_source

        datasets: dict[str, CalibrationDatasetMeta] = {}
        returned_data: dict = {}

        for name in dataset_order:
            ds_payload = dataset_payloads.get(name)
            prefix = dataset_prefixes.get(name)
            if ds_payload is None or prefix is None:
                raise ValueError(f'Invalid dataset entry in manifest for {name}.')

            datasets[name] = dataset_from_payload(ds_payload)

            params = {}
            for key in metric_keys.get(name, []):
                array_key = f'{prefix}__{key}'
                if array_key not in payload:
                    continue

                value = np.array(payload[array_key])
                if value.ndim == 0:
                    scalar = value.item()
                    if isinstance(scalar, np.generic):
                        scalar = scalar.item()
                    params[key] = scalar
                else:
                    params[key] = value

            if 'fit_variance_source' not in params:
                params['fit_variance_source'] = resolved_gain_variance_source

            if 'fit_variance_dn2' not in params:
                if (
                    params['fit_variance_source'] == 'total'
                    and 'var_total_dn2' in params
                ):
                    params['fit_variance_dn2'] = params['var_total_dn2']
                elif 'var_shot_dn2' in params:
                    params['fit_variance_dn2'] = params['var_shot_dn2']

            returned_data[name] = params

        exposure_times_s = np.array(payload['exposure_times_s'])

    return datasets, returned_data, exposure_times_s, analysis_options
