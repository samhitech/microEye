import os
import platform
from pathlib import Path

try:
    import vmbpy as vb
except Exception:
    vb = None

IS_VIMBA_PATH_SET = False

def _resolve_vimba_home() -> list[str] | None:
    cti = []

    manual = os.environ.get('VIMBA_X_CTI')
    if manual:
        paths = manual.split(os.pathsep)
        for path in paths:
            if os.path.exists(path):
                cti.append(Path(path))
        if cti:
            return cti

    vimba_home = os.environ.get('VIMBA_X_HOME')
    if vimba_home:
        home_path = os.path.join(vimba_home, 'cti')
        if os.path.exists(home_path):
            return [home_path]

    system = platform.system()
    candidates = []

    if system == 'Windows':
        program_files = Path(os.environ.get('PROGRAMFILES', r'C:\Program Files'))
        candidates = [
            program_files / 'Allied Vision' / 'Vimba X' / 'cti',
            Path('C:/Allied Vision/Vimba X/cti'),
            Path('D:/Allied Vision/Vimba X/cti'),
        ]
    elif system == 'Linux':
        candidates = [
            Path('/opt/VimbaX/cti'),
            Path('/usr/local/VimbaX/cti'),
        ]
    elif system == 'Darwin':
        candidates = [
            Path('/Applications/Vimba X.app/Contents/cti'),
            Path('/Applications/VimbaX/cti'),
        ]

    for candidate in candidates:
        if candidate.exists():
            return [str(candidate)]

    return cti

def get_instance():
    instance = vb.VmbSystem.get_instance()

    global IS_VIMBA_PATH_SET

    if not IS_VIMBA_PATH_SET:
        path = _resolve_vimba_home()
        if path:
            instance.set_path_configuration(*path)
            IS_VIMBA_PATH_SET = True

    return instance
