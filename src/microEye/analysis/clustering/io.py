import json
from dataclasses import asdict
from enum import Enum

import h5py
import numpy as np

from microEye.analysis.clustering.model import InstanceStats

_EXCLUDE_FROM_RUN_META = {'save_results', 'save_suffix', 'reference_lines'}

def serialize_kwargs(kwargs: dict) -> dict:
    safe = {}
    for k, v in kwargs.items():
        if k in _EXCLUDE_FROM_RUN_META:
            continue
        if isinstance(v, Enum):
            safe[k] = v.value
        elif isinstance(v, (np.integer, np.floating)):
            safe[k] = v.item()
        elif isinstance(v, np.ndarray):
            safe[k] = v.tolist()
        elif isinstance(v, (str, int, float, bool)) or v is None:
            safe[k] = v
        elif isinstance(v, (list, dict)):
            try:
                json.dumps(v)
                safe[k] = v
            except TypeError:
                continue
        # else: silently dropped (e.g. FittingResults objects, functions)
    return safe

def export_run(
    tables: list[tuple[dict, 'InstanceStats']],
    filepath: str,
    model_meta: dict,
    run_meta=None
):
    '''
    Save a batch of aligned tables and their stats to an HDF5 file.

    Parameters
    ----------
    tables : list of (aligned, stats) tuples
        - Each aligned is a dict of {cluster_id: data_dict}, where data_dict contains
        arrays of x, y, z, intensity, labels, crlb_x, crlb_y, etc.
        - Each stats is an `InstanceStats` object.
    filepath : str
        Path to the output HDF5 file.
    model_meta : dict
        Metadata about the model used for fitting (e.g., n_sites, expected_spacing).
    run_meta : dict, optional
        Metadata about the run (e.g., parameters used). If not provided, will be
        inferred from the current function's kwargs, excluding certain keys.
    '''
    if run_meta is None:
        run_meta = {}

    with h5py.File(filepath, 'w') as f:
        f.attrs['file_type'] = 'microEye.analysis.clustering.model'
        f.attrs['model_meta'] = json.dumps(model_meta)
        f.attrs['run_meta'] = json.dumps(serialize_kwargs(run_meta))
        f.attrs['n_tables'] = len(tables)
        for t_idx, (aligned, stats) in enumerate(tables):
            tg = f.create_group(f'table_{t_idx}')
            tg.attrs['stats'] = json.dumps(asdict(stats))
            for gid, data in aligned.items():
                g = tg.create_group(str(gid))
                for key, val in data.items():
                    if val is not None:
                        g.create_dataset(key, data=np.asarray(val))


def load_run(filepath: str) -> tuple[list[dict], list[dict], dict]:
    '''
    Load a batch of aligned tables and their stats from an HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to the input HDF5 file.

    Returns
    -------
    tables_aligned : list of dict
        Each dict is {cluster_id: data_dict}, where data_dict contains arrays of
        x, y, z, intensity, labels, crlb_x, crlb_y, etc.
    tables_stats : list of dict
        Each dict is the serialized form of an `InstanceStats` object.
    model_meta : dict
        Metadata about the model used for fitting (e.g., n_sites, expected_spacing).
    run_meta : dict
        Metadata about the run (e.g., parameters used).
    '''
    tables_aligned, tables_stats = [], []
    with h5py.File(filepath, 'r') as f:
        model_meta = json.loads(f.attrs['model_meta'])
        run_meta = json.loads(f.attrs.get('run_meta', '{}'))
        for t_idx in range(f.attrs['n_tables']):
            tg = f[f'table_{t_idx}']
            tables_stats.append(json.loads(tg.attrs['stats']))
            aligned = {gid_str: {k: g[k][()] for k in g} for gid_str, g in tg.items()}
            tables_aligned.append({int(k): v for k, v in aligned.items()})
    return tables_aligned, tables_stats, model_meta, run_meta
