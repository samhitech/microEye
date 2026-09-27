import argparse
import importlib.util
import logging
import os

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

from microEye._version import __version__, version_info

# QT bindings affect camera libraries' DLL loading.
# Hence, we need to import camera libraries at package initialization.
# Before the QT is imported or QT event loop is started.
from microEye.hardware import cams

PREFERRED_QT_APIS = ('PySide6', 'PyQt6', 'PyQt5')

HELP_MESSAGE = ''


def _select_qt_api(cli_choice: str | None) -> str:
    candidates = (cli_choice,) + PREFERRED_QT_APIS if cli_choice else PREFERRED_QT_APIS
    for api in candidates:
        if api and importlib.util.find_spec(api):
            os.environ.setdefault('QT_API', api)
            os.environ.setdefault('PYQTGRAPH_QT_LIB', api)
            return api
    raise ImportError('Missing Qt packages, install one of PySide6, PyQt6, PyQt5.')


def getArgs():
    import os
    import re
    import sys

    parser = argparse.ArgumentParser(
        add_help=False,
    )
    parser.add_argument(
        '-h', '--help', action='store_true', help='Show this help message and exit.'
    )
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

    lines = []
    for line in parser.format_help().splitlines():
        # format flags (cyan)
        line = re.sub(r'(?<=\s|\[)(-\S+)', r'[bold #00d7ff]\1[/bold #00d7ff]', line)
        line = re.sub(r'(?<=\s|\[)(--\S+)', r'[bold #00d7ff]\1[/bold #00d7ff]', line)

        # format scripts & files (warm yellow)
        line = re.sub(r'(?<=\s)(\S+\.py)', r'[#ffb86c]\1[/#ffb86c]', line)
        line = re.sub(r'(?<=\s)(\S+\.exe)', r'[#ffb86c]\1[/#ffb86c]', line)
        line = re.sub(r'(?<=\s)(\S+\.dll)', r'[#ffb86c]\1[/#ffb86c]', line)
        line = re.sub(r'(?<=\s)(\S+\.so)', r'[#ffb86c]\1[/#ffb86c]', line)
        line = re.sub(r'(?<=\s)(\S+\.dylib)', r'[#ffb86c]\1[/#ffb86c]', line)

        # format curly braces (bright green)
        line = re.sub(r'(?<=\s)(\{.*?\})', r'[bold #5af78e]\1[/bold #5af78e]', line)

        # format flags within brackets (cyan)
        line = re.sub(r'(?<=\[)(--\S+)', r'[bold #00d7ff]\1[/bold #00d7ff]', line)
        line = re.sub(r'(?<=\[)(-\S+)', r'[bold #00d7ff]\1[/bold #00d7ff]', line)

        line = line.replace('usage:', '[bold #dcdcaa]usage:[/bold #dcdcaa]')
        line = line.replace('options:', '[bold #dcdcaa]options:[/bold #dcdcaa]')

        lines.append(line)

    global HELP_MESSAGE
    HELP_MESSAGE = '\n'.join(lines) + '\n'

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

    if not args.no_log_file and args.module:
        console = Console(
            file=open(log_filename, 'a', encoding='utf-8')  # noqa: SIM115, W291
        )
        file_handler = RichHandler(
                        console=console,
                        rich_tracebacks=True,
                        tracebacks_show_locals=True,
                        markup=True,
                    )
    else:
        file_handler = logging.NullHandler()

    # set the logging configuration with both file and console handlers
    logging.basicConfig(
        level=level,
        format='%(message)s',
        handlers=[
            file_handler,
            RichHandler(
                rich_tracebacks=True,
                tracebacks_show_locals=True,
                markup=True,
            )
            if not args.no_log_console
            else logging.NullHandler(),
        ],
        force=True,
    )
    logging.getLogger().debug('Log level set to %s', level_name)


install(
    show_locals=False,
    code_width=150,
    width=200,
    extra_lines=2,
)
ARGS = getArgs()
