import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, Union

import cv2
import h5py
import numba as nb
import numpy as np
from tqdm import tqdm

from microEye.analysis.fitting.results import PARAMETER_HEADERS
from microEye.utils.uImage import TiffSeqHandler, ZarrImageSequence, uImage


class PSFdata:
    def __init__(
        self, data_dict: dict[str, Union[int, np.ndarray, list, float]]
    ) -> None:
        self._data = data_dict

    def get_field_psf(self, grid_size: int = 3):
        '''
        Get PSF field for the given grid size.
        '''
        width, height = self.dim
        width *= self.upsample
        height *= self.upsample
        offset = self.roi_size / 2

        def get_grid_cell_index(x_start, y_start):
            if not np.isfinite(x_start) or not np.isfinite(y_start):
                return -1
            cell_width = width // grid_size
            cell_height = height // grid_size
            i = (y_start + offset) // cell_height
            j = (x_start + offset) // cell_width
            return int(i * grid_size + j)

        for zslice in self._data['zslices']:
            zslice['field'] = [[] for _ in range(grid_size**2)]
            zslice['field_idx'] = np.zeros(zslice['rois'].shape[0], dtype=np.uint16)
            for i in range(zslice['rois'].shape[0]):
                idx = get_grid_cell_index(*zslice['coords'][i])
                if idx >= 0:
                    zslice['field'][idx].append(zslice['rois'][i].copy())
                zslice['field_idx'][i] = idx

    @property
    def zslices(self) -> list[dict[str, Union[int, np.ndarray, list, float]]]:
        return self._data.get('zslices')

    @property
    def fitting_method(self) -> float:
        return self._data.get('fit_method')

    @property
    def pixel_size(self) -> float:
        return self._data.get('pixel_size')

    @property
    def z_step(self) -> float:
        step = self._data.get('z_step', 10)
        if step:
            return step
        else:
            return 10

    @property
    def roi_size(self) -> int:
        return self._data.get('roi_size')

    @property
    def roi_info(self):
        return self._data.get('roi_info')

    @property
    def origin(self):
        if self.roi_info:
            return self.roi_info[0]
        else:
            return (0, 0)

    @property
    def dim(self):
        if self.roi_info:
            return tuple(self.roi_info[1])
        else:
            return self.shape[-1], self.shape[-2]

    @property
    def upsample(self) -> int:
        return self._data.get('upsample')

    @property
    def zero_plane(self) -> Optional[int]:
        return self._data.get('zero_plane')

    @zero_plane.setter
    def zero_plane(self, value: int):
        self._data['zero_plane'] = value

    @property
    def shape(self) -> tuple:
        return self._data.get('shape')

    @property
    def path(self) -> str:
        return self._data.get('stack')

    @property
    def rois(self):
        return [zslice['rois'] for zslice in self.zslices]

    @property
    def counts(self) -> np.ndarray:
        return np.array([zslice['count'] for zslice in self.zslices])

    @property
    def mean(self) -> np.ndarray:
        return np.array([zslice['mean'] for zslice in self.zslices])

    @property
    def median(self) -> np.ndarray:
        return np.array([zslice['median'] for zslice in self.zslices])

    @property
    def std(self) -> np.ndarray:
        return np.array([zslice['std'] for zslice in self.zslices])

    @property
    def coords(self) -> np.ndarray:
        return np.array([zslice['coords'] for zslice in self.zslices])

    @property
    def params(self) -> np.ndarray:
        return np.array([zslice['params'] for zslice in self.zslices])

    @property
    def crlbs(self) -> np.ndarray:
        return np.array([zslice['crlbs'] for zslice in self.zslices])

    @property
    def loglike(self) -> np.ndarray:
        return np.array([zslice['loglike'] for zslice in self.zslices])

    @property
    def field_rois(self) -> list:
        return [zslice.get('field') for zslice in self.zslices]

    @property
    def field_indecies(self) -> np.ndarray:
        return np.array([zslice['field_idx'] for zslice in self.zslices])

    def __getitem__(self, key: str) -> Any:
        return self._data.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __repr__(self) -> str:
        return f'PSFdata({self._data})'

    def __str__(self) -> str:
        return f'PSFdata({self._data})'

    def __len__(self) -> int:
        return len(self.zslices)

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self.zslices):
            result = self.zslices[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration

    def get_consistent_rois(self) -> 'PSFdata':
        '''
        Creates a new PSFdata object with consistent ROIs across all z-slices.
        Only keeps coordinates that exist in all slices and ensures consistent ordering.

        Returns
        -------
        PSFdata
            A new PSFdata object with consistent ROIs
        '''
        # Create a copy of the original data
        new_data = self._data.copy()

        # Tolerance for matching coordinates
        tolerance = 2.0 * self.upsample

        # First, identify coordinates that exist in all slices
        all_coords = []
        for zslice in self.zslices:
            params = zslice['params']
            # Only consider non-NaN coordinates
            valid_coords = params[~np.isnan(params[:, 0])][:, :2]
            all_coords.append(valid_coords)

        # Function to find matching coordinates within a tolerance
        def find_matching_coords(coord, coords_list):
            matches = []
            for slice_idx, slice_coords in enumerate(coords_list):
                distances = np.sqrt(np.sum((slice_coords - coord) ** 2, axis=1))
                match_idx = np.argmin(distances)
                if distances[match_idx] <= tolerance:
                    matches.append((slice_idx, match_idx))
            return matches if len(matches) == len(coords_list) else None

        # Find coordinates that exist in all slices
        consistent_coords = []
        consistent_indices = [[] for _ in range(len(all_coords))]

        for _, coord in enumerate(all_coords[0]):
            matches = find_matching_coords(coord, all_coords)
            if matches:
                consistent_coords.append(coord)
                for slice_idx, match_idx in matches:
                    consistent_indices[slice_idx].append(match_idx)

        # Create new zslices with only consistent ROIs
        new_data['zslices'] = []
        for slice_idx, zslice in enumerate(self.zslices):
            indices = consistent_indices[slice_idx]

            new_zslice = {
                'index': zslice['index'],
                'rois': zslice['rois'][indices] if zslice['rois'] is not None else None,
                'coords': zslice['coords'][indices]
                if zslice['coords'] is not None
                else None,
                'params': zslice['params'][indices],
                'crlbs': zslice['crlbs'][indices],
                'loglike': zslice['loglike'][indices],
                'count': len(indices),
            }

            # Recalculate statistics for the consistent ROIs
            if new_zslice['rois'] is not None:
                new_zslice['mean'] = np.nanmean(new_zslice['rois'], axis=0)
                new_zslice['median'] = np.nanmedian(new_zslice['rois'], axis=0)
                new_zslice['std'] = np.nanstd(new_zslice['rois'], axis=0)
            else:
                new_zslice['mean'] = None
                new_zslice['median'] = None
                new_zslice['std'] = None

            new_data['zslices'].append(new_zslice)

        return PSFdata(new_data)

    def save_hdf(self, filename: str):
        '''
        Save PSF data to HDF5 file.
        '''
        with h5py.File(filename, 'w') as hdf:
            # Save scalar attributes
            for key in ['pixel_size', 'roi_size', 'upsample', 'z_step']:
                hdf.attrs[key] = self._data[key]

            hdf.attrs['shape'] = json.dumps(self.shape)
            hdf.attrs['stack'] = self.path
            hdf.attrs['fit_method'] = self.fitting_method
            hdf.attrs['zero_plane'] = self.zero_plane
            hdf.attrs['roi_info'] = json.dumps(self.roi_info)

            # Save zslices data
            zslices_group = hdf.create_group('zslices')
            for i, zslice in enumerate(self.zslices):
                slice_group = zslices_group.create_group(f'slice_{i}')
                for key, value in zslice.items():
                    if isinstance(value, np.ndarray):
                        slice_group.create_dataset(key, data=value)
                    elif value is None:
                        slice_group.attrs[key] = 'None'
                    else:
                        slice_group.attrs[key] = value

        print(f'PSF data saved to {filename}')

    @staticmethod
    def load_hdf(filename: str):
        '''
        Load PSF data from HDF5 file.
        '''
        if not os.path.exists(filename):
            raise FileNotFoundError(f'File not found: {filename}')

        with h5py.File(filename, 'r') as hdf:
            data = {
                'zslices': [],
            }

            for key in [
                'pixel_size',
                'z_step',
                'roi_size',
                'upsample',
                'stack',
                'fit_method',
                'zero_plane',
            ]:
                data[key] = hdf.attrs.get(key, None)

            for key in ['shape', 'roi_info', 'params_headers']:
                data[key] = json.loads(hdf.attrs.get(key, '[]'))

            zslices_group = hdf['zslices']
            # Sort slice names numerically
            slice_names = sorted(
                zslices_group.keys(), key=lambda x: int(x.split('_')[1])
            )

            for slice_name in slice_names:
                slice_group = zslices_group[slice_name]
                zslice_data = {}
                for key in slice_group:
                    zslice_data[key] = slice_group[key][()]
                for key in slice_group.attrs:
                    if slice_group.attrs[key] == 'None':
                        zslice_data[key] = None
                    else:
                        zslice_data[key] = slice_group.attrs[key]
                data['zslices'].append(zslice_data)

            # Validate required attributes
            required_attrs = ['pixel_size', 'roi_size', 'upsample', 'shape', 'stack']
            missing_attrs = [attr for attr in required_attrs if attr not in data]
            if missing_attrs:
                raise ValueError(f'Missing required attributes: {missing_attrs}')

        return PSFdata(data)


@nb.njit(cache=True)
def get_roi_list(image: np.ndarray, points: np.ndarray, roi_size=7):
    '''
    Gets the roi list of specific size around the supplied (x, y) points

    Parameters
    ----------
    image : np.ndarray
        The single channel image
    points : np.ndarray
        The points list of preliminary detection
    roi_size : int, optional
        roi size, by default 7

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        roi_list array of shape (nRoi, roi_size**2),
        coord_list of roi top left corner
    '''
    if len(points) < 1:
        return None

    assert len(image.shape) == 2, 'image should be a 2D ndarray!'

    roi_list = np.zeros((points.shape[0], roi_size, roi_size), np.float32)
    coord_list = np.zeros_like(points)

    half_size = roi_size // 2
    y_max, x_max = image.shape

    for r in nb.prange(points.shape[0]):
        x, y = points[r, :]

        x_start = int(x - half_size)
        y_start = int(y - half_size)
        x_end = x_start + roi_size
        y_end = y_start + roi_size

        # Ensure ROI is within image bounds
        # if x_start < 0:
        #     x_start = 0
        #     x_end = roi_size
        # elif x_end > x_max:
        #     x_end = x_max
        #     x_start = x_end - roi_size

        # if y_start < 0:
        #     y_start = 0
        #     y_end = roi_size
        # elif y_end > y_max:
        #     y_end = y_max
        #     y_start = y_end - roi_size

        if x_start < 0 or x_end > x_max or y_start < 0 or y_end > y_max:
            coord_list[r, :] = [np.nan, np.nan]
            roi_list[r] = np.nan
        else:
            coord_list[r, :] = [x_start, y_start]
            roi_list[r] = image[y_start:y_end, x_start:x_end]

    return roi_list, coord_list


def find_best_z(point_params: np.ndarray, z0_criteria: str, headers: list[str]):
    '''
    Find the best z0 index based on the given criteria.
    '''
    z0_criteria = z0_criteria.lower().replace(' ', '_')

    def get_column(header: str):
        return point_params[:, headers.index(header)]

    if z0_criteria == 'intensity':
        intensities = get_column('intensity')
        return np.argmax(intensities)
    elif z0_criteria == 'min_sigma' and 'sigmax' in headers:
        sigmax = get_column('sigmax')
        return np.argmin(sigmax)
    elif z0_criteria == 'min_sum_sigma' and 'sigmax' in headers and 'sigmay' in headers:
        sigmax = get_column('sigmax')
        sigmay = get_column('sigmay')
        return np.argmin(sigmax + sigmay)
    else:
        argmins = []
        if 'sigmax' in headers and 'sigmay' in headers:
            sigmax = get_column('sigmax')
            sigmay = get_column('sigmay')
            intensities = get_column('intensity')
            argmins.append(np.argmin(sigmax + sigmay))
            argmins.append(np.argmax(intensities))
        else:
            argmins.append(np.argmax(get_column('intensity')))
            if 'sigmax' in headers:
                argmins.append(np.argmin(get_column('sigmax')))

        # return average index
        return int(np.mean(argmins))


def get_psf_rois(
    stack_handler: Union[TiffSeqHandler, ZarrImageSequence],
    frame_list,
    params,
    crlbs,
    loglike,
    fit_method: int,
    pixel_size: float = 114.17,
    z_step: float = 10,
    roi_size: int = 13,
    upsample: int = 1,
    roi_info: Optional[tuple] = None,
    find_z0: bool = False,
    z0_criteria: str = 'all',
) -> 'PSFdata':
    '''
    Get PSF ROIs for the given frame list.
    '''
    if upsample > 1:
        roi_size = roi_size * upsample | 1
        # This uses the bitwise OR operator to ensure the least significant bit is
        # set to 1, making the number odd.
    else:
        upsample = 1

    headers = PARAMETER_HEADERS[fit_method]
    frame_ids = np.cumsum(np.bincount(np.array(frame_list, np.int64) - 1))
    frames = np.arange(0, max(frame_list), 1, dtype=np.uint32)

    if find_z0:
        frame_params = np.zeros((len(frames), len(headers)))
        for i in range(len(frames)):
            start_idx = frame_ids[i - 1] if i > 0 else 0
            end_idx = frame_ids[i]
            frame_params[i] = np.nanmean(params[slice(start_idx, end_idx)], axis=0)

        z0 = find_best_z(frame_params, z0_criteria, headers)
    else:
        z0 = max(frame_list) // 2

    def process_frame(index: int) -> dict[str, Union[int, np.ndarray, list, float]]:
        image = stack_handler.getSlice(index, 0, 0)

        if roi_info is not None:
            origin, dim = roi_info
            slice_y = slice(int(origin[1]), int(origin[1] + dim[1]))
            slice_x = slice(int(origin[0]), int(origin[0] + dim[0]))
            image = image[slice_y, slice_x]
        else:
            origin = (0, 0)
            dim = (image.shape[1], image.shape[0])

        if upsample > 1:
            image = cv2.resize(
                image,
                (0, 0),
                fx=upsample,
                fy=upsample,
                interpolation=cv2.INTER_NEAREST,
            )

        param = params[
            slice(frame_ids[index - 1] if index > 0 else 0, frame_ids[index])
        ]
        crlb = crlbs[slice(frame_ids[index - 1] if index > 0 else 0, frame_ids[index])]
        logl = loglike[
            slice(frame_ids[index - 1] if index > 0 else 0, frame_ids[index])
        ]

        if z0 is None:
            points = upsample * (param[..., :2] - origin) + (upsample - 1) / 2
        else:
            points = (
                upsample
                * (
                    params[
                        slice(frame_ids[z0 - 1] if z0 > 0 else 0, frame_ids[z0]),
                        :2,
                    ]
                    - origin
                )
                + (upsample - 1) / 2
            )

        res = get_roi_list(image, points, roi_size)

        if res:
            rois, coords = res
            count = rois.shape[0]
        else:
            rois, coords = None, None
            count = 0

        return {
            'index': index,
            'rois': rois,
            'count': count,
            'mean': np.nanmean(rois, axis=0) if count else None,
            'median': np.nanmedian(rois, axis=0) if count else None,
            'std': np.nanstd(rois, axis=0) if count else None,
            'coords': coords,
            'params': param,
            'crlbs': crlb,
            'loglike': logl,
        }

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_frame, i) for i in frames]
        results = [None] * len(frames)
        for future in tqdm(
            as_completed(futures), total=len(futures), desc='Extrating PSF ROIs'
        ):
            idx = futures.index(future)
            results[idx] = future.result()

    return PSFdata(
        {
            'zslices': results,
            'shape': stack_handler.shapeTCZYX(),
            'pixel_size': pixel_size,
            'roi_size': roi_size,
            'upsample': upsample,
            'stack': stack_handler.path,
            'roi_info': roi_info,
            'fit_method': fit_method,
            'zero_plane': z0,
            'z_step': z_step,
        }
    )
