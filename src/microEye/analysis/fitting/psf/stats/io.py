import json
import os
from dataclasses import asdict
from typing import Optional, Union

from microEye.analysis.fitting.psf.stats.curve_fit import CurveFitMethod, CurveResult
from microEye.analysis.fitting.psf.stats.slope_fit import SlopeResult
from microEye.qt import QtWidgets, getOpenFileName, getSaveFileName
from microEye.utils.enum_encoder import EnumEncoder


def import_fit_curve(parent: Optional[QtWidgets.QWidget] = None, curDir: str = ''):
    '''
    Import curve fit results from a JSON file.

    Parameters
    ----------
    parent : Optional[QtWidgets.QWidget]
        Parent widget for file dialog
    curDir : str
        Initial directory for file dialog

    Notes
    -----
    This function requires a QApplication instance to be running

    Returns
    -------
    Tuple[Union[SlopeResult, CurveResult], str]
        The curve fit results and the
        filename from which they were imported
    '''
    filename, _ = getOpenFileName(
        parent,
        'Open Curve Fit',
        curDir,
        'JSON Files (*.json)',
    )

    if not filename:
        return None, None

    with open(filename) as file:
        data = json.load(file)

    if 'method' not in data:
        return None, None

    try:
        data['method'] = CurveFitMethod(data['method'])

        if data['method'] == CurveFitMethod.LINEAR:
            return SlopeResult(**data), filename
        else:
            return CurveResult(**data), filename
    except ValueError:
        # Value is not a valid enum member; keep as is
        return None, None


def export_fit_curve(
    results: Union[SlopeResult, CurveResult],
    parent: Optional[QtWidgets.QWidget] = None,
    curDir: str = '',
):
    '''
    Export curve fit results to a JSON file.

    Parameters
    ----------
    results : Union[SlopeResult, CurveResult]
        The curve fit results to export
    parent : Optional[QtWidgets.QWidget]
        Parent widget for file dialog
    curDir : str
        Initial directory for file dialog

    Notes
    -----
    This function requires a QApplication instance to be running

    Returns
    -------
    bool
        True if the results were successfully exported
    '''
    if results is None:
        return False

    filename, _ = getSaveFileName(
        parent,
        'Save Curve Fit',
        curDir,
        'JSON Files (*.json)',
    )

    if not filename:
        return False

    if isinstance(results, (SlopeResult, CurveResult)):
        with open(filename, 'w') as file:
            json.dump(asdict(results), file, cls=EnumEncoder)

        return True
