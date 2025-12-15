import importlib.util
import logging
import os

from microEye._version import __version__, version_info

# QT bindings affect camera libraries' DLL loading.
# Hence, we need to import camera libraries at package initialization.
# Before the QT is imported or QT event loop is started.
from microEye.hardware import cams

PREFERRED_QT_APIS = ('PySide6', 'PyQt6', 'PyQt5')


def _select_qt_api(cli_choice: str | None) -> str:
    candidates = (cli_choice,) + PREFERRED_QT_APIS if cli_choice else PREFERRED_QT_APIS
    for api in candidates:
        if api and importlib.util.find_spec(api):
            os.environ.setdefault('QT_API', api)
            os.environ.setdefault('PYQTGRAPH_QT_LIB', api)
            return api
    raise ImportError('Missing Qt packages, install one of PySide6, PyQt6, PyQt5.')


def getArgs():
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--module',
        help='The module to launch [mieye|viewer], '
        + 'If not specified, launcher is executed.',
        choices=['mieye', 'viewer'],
    )
    parser.add_argument(
        '--QT_API',
        help='Select QT API [PySide6|PyQT6|PyQt5], '
        + 'If not specified, the environment variable QT_API is used.',
        choices=['PySide6', 'PyQt6', 'PyQt5'],
    )
    parser.add_argument(
        '--theme',
        help='The theme of the app, '
        + 'if not specified, the environment variable MITHEME is used.',
        choices=['None', 'qdarktheme', 'qdarkstyle', '...'],
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
        help='Root logging level (default INFO).',
    )
    parser.add_argument(
        '--no-log-file',
        help='Disable logging to file.',
        action='store_true',
    )
    parser.add_argument(
        '--no-log-console',
        help='Disable logging to console.',
        action='store_true',
    )

    args, _ = parser.parse_known_args(sys.argv)

    # access the parsed arguments
    qt_api = args.QT_API or os.environ.get('QT_API')
    _select_qt_api(qt_api)

    os.environ.setdefault('MITHEME', args.theme or 'qdarktheme')

    configure_logger(args)

    return args

def configure_logger(args):
    import os
    import sys

    level_name = os.environ.get('MICROEYE_LOG_LEVEL', args.log_level).upper()
    level = getattr(logging, level_name, logging.INFO)

    # create file with current date time stamp
    log_filename = os.path.join(
        os.getcwd(),
        f'logs\\microEye_log_{__version__}_{os.getpid()}_.log',
    )
    if (
        not os.path.exists(os.path.dirname(log_filename))
        and not args.no_log_file
        and args.module
    ):
        os.makedirs(os.path.dirname(log_filename))
    # set the logging configuration with both file and console handlers
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_filename)
            if not args.no_log_file and args.module
            else logging.NullHandler(),
            logging.StreamHandler(sys.stdout)
            if not args.no_log_console
            else logging.NullHandler(),
        ],
        force=True,
    )
    logging.getLogger().debug('Log level set to %s', level_name)



ARGS = getArgs()
