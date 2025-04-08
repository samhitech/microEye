# get pycromanager version
from mmpycorex import __version__ as mmpycorex_version
from pycromanager import __version__ as pycromanager_version

required_pycromanager_version = '1.0.2'
required_mmpycorex_version = '0.3.16'

if pycromanager_version != required_pycromanager_version:
    raise ImportError(
        f'This module requires pycromanager version {required_pycromanager_version}'
    )

if mmpycorex_version < required_mmpycorex_version:
    raise ImportError(
        f'This module requires mmpycorex version {required_mmpycorex_version}'
    )
