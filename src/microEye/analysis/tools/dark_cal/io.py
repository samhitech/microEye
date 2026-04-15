import json
import os
from datetime import datetime, timezone

import numpy as np

from microEye import __version__
from microEye.analysis.tools.common import (
    normalize_positive_float,
    to_bool,
    to_float_or_none,
)
from microEye.analysis.tools.dark_cal.constants import (
    FILE_NAMES,
    HISTOGRAM_DATA_TYPES,
    RESULTS_SCHEMA_VERSION,
    DataTypes,
)


def get_file_stats(file_path: str):
    if not os.path.exists(file_path):
        return None

    stat = os.stat(file_path)
    mtime_ns = getattr(stat, 'st_mtime_ns', int(stat.st_mtime * 1e9))

    return {
        'size': int(stat.st_size),
        'mtime_ns': int(mtime_ns),
    }


def get_source_files_metadata(directory: str) -> dict:
    metadata = {}

    for _data_type, file_name in FILE_NAMES.items():
        file_path = os.path.join(directory, file_name)
        stats = get_file_stats(file_path)
        if stats is not None:
            metadata[file_name] = stats

    return metadata


def _default_dataset_meta(directory: str) -> dict:
    base_name = os.path.basename(os.path.normpath(str(directory))) or str(directory)
    return {
        'name': base_name,
        'dark_calibration_directory': str(directory),
        'gain': 1.0,
        'gain_defaulted': True,
        'responsivity': None,
        'quantum_efficiency': None,
    }


def _serialize_dataset_meta(directory: str, dataset_meta: dict | None) -> dict:
    default_meta = _default_dataset_meta(directory)
    source = dataset_meta or {}

    gain, gain_defaulted = normalize_positive_float(source.get('gain'), default=1.0)
    if not gain_defaulted:
        gain_defaulted = to_bool(source.get('gain_defaulted'), default=False)

    return {
        'name': str(source.get('name', default_meta['name'])),
        'dark_calibration_directory': str(
            source.get('dark_calibration_directory', directory)
        ),
        'gain': gain,
        'gain_defaulted': gain_defaulted,
        'responsivity': to_float_or_none(source.get('responsivity')),
        'quantum_efficiency': to_float_or_none(source.get('quantum_efficiency')),
    }


def _deserialize_dataset_meta(directory: str, payload: dict | None) -> dict:
    default_meta = _default_dataset_meta(directory)
    source = payload or {}

    gain, gain_defaulted = normalize_positive_float(source.get('gain'), default=1.0)
    if not gain_defaulted:
        gain_defaulted = to_bool(source.get('gain_defaulted'), default=False)

    return {
        'name': str(source.get('name', default_meta['name'])),
        'dark_calibration_directory': str(
            source.get('dark_calibration_directory', directory)
        ),
        'gain': gain,
        'gain_defaulted': gain_defaulted,
        'responsivity': to_float_or_none(source.get('responsivity')),
        'quantum_efficiency': to_float_or_none(source.get('quantum_efficiency')),
    }


def save_results_npz(
    file_path: str,
    directories: dict,
    mode: str,
    dataset_meta: dict[str, dict] | None = None,
) -> int:
    arrays = {}
    manifest = {
        'schema_version': RESULTS_SCHEMA_VERSION,
        'tool': 'DarkCalibration',
        'microeye_version': __version__,
        'mode': mode,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'datasets': [],
    }

    for index, (directory, data) in enumerate(directories.items()):
        prefix = f'd{index:04d}'
        dataset_entry = {
            'id': prefix,
            'directory': directory,
            'source_files': get_source_files_metadata(directory),
            'dataset_meta': _serialize_dataset_meta(
                directory,
                None if dataset_meta is None else dataset_meta.get(directory),
            ),
        }

        if mode == 'Standard':
            arrays[f'{prefix}_exposure'] = np.asarray(data[DataTypes.EXPOSURE])
            arrays[f'{prefix}_mean'] = np.asarray(data[DataTypes.MEAN])
            arrays[f'{prefix}_variance'] = np.asarray(data[DataTypes.VARIANCE])

            temp = data.get(DataTypes.TEMPERATURE)
            has_temperature = temp is not None
            dataset_entry['has_temperature'] = has_temperature
            if has_temperature:
                arrays[f'{prefix}_temperature'] = np.asarray(temp)
        else:
            for data_type in HISTOGRAM_DATA_TYPES:
                hist_data = data[data_type]
                root = f'{prefix}_{data_type.value}'
                arrays[f'{root}_hist'] = np.asarray(hist_data['hist'])
                arrays[f'{root}_bin_edges'] = np.asarray(hist_data['bin_edges'])
                arrays[f'{root}_median'] = np.asarray(
                    hist_data['median'], dtype=np.float64
                )
                arrays[f'{root}_mad'] = np.asarray(hist_data['mad'], dtype=np.float64)

        manifest['datasets'].append(dataset_entry)

    arrays['manifest'] = np.array(json.dumps(manifest, indent=2))
    np.savez(file_path, **arrays)
    return len(manifest['datasets'])


def load_results_npz(file_path: str) -> tuple[str, dict, dict]:
    directories = {}
    loaded_meta = {}

    with np.load(file_path, allow_pickle=False) as payload:
        if 'manifest' not in payload:
            raise ValueError('Missing manifest in NPZ file.')

        manifest = json.loads(str(np.array(payload['manifest']).item()))
        schema_version = int(manifest.get('schema_version', 0))
        if schema_version > RESULTS_SCHEMA_VERSION:
            raise ValueError(
                f'Unsupported schema version {schema_version}. '
                f'Max supported is {RESULTS_SCHEMA_VERSION}.'
            )

        mode = manifest.get('mode')
        if mode not in ('Standard', 'Histograms'):
            raise ValueError(f'Unsupported analysis mode in cache: {mode}')

        for entry in manifest.get('datasets', []):
            prefix = entry.get('id')
            directory = entry.get('directory')
            if not prefix or directory is None:
                raise ValueError('Invalid dataset entry in manifest.')

            if mode == 'Standard':
                data = {
                    DataTypes.EXPOSURE: np.array(payload[f'{prefix}_exposure']),
                    DataTypes.MEAN: np.array(payload[f'{prefix}_mean']),
                    DataTypes.VARIANCE: np.array(payload[f'{prefix}_variance']),
                    DataTypes.TEMPERATURE: None,
                }
                temp_key = f'{prefix}_temperature'
                if temp_key in payload:
                    data[DataTypes.TEMPERATURE] = np.array(payload[temp_key])
            else:
                data = {}
                for data_type in HISTOGRAM_DATA_TYPES:
                    root = f'{prefix}_{data_type.value}'
                    data[data_type] = {
                        'hist': np.array(payload[f'{root}_hist']),
                        'bin_edges': np.array(payload[f'{root}_bin_edges']),
                        'median': float(np.array(payload[f'{root}_median']).item()),
                        'mad': float(np.array(payload[f'{root}_mad']).item()),
                    }

            directory_str = str(directory)
            directories[directory_str] = data
            loaded_meta[directory_str] = _deserialize_dataset_meta(
                directory_str,
                entry.get('dataset_meta'),
            )

    return mode, directories, loaded_meta
