from microEye._version import __version__, version_info


def getArgs():
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--module',
        help='The module to launch [mieye|viewer], '
        + 'If not specified, launcher is executed.',
        choices=['mieye', 'viewer']
    )
    parser.add_argument(
        '--QT_API',
        help='Select QT API [PySide6|PyQT6|PyQt5], '
        + 'If not specified, the environment variable QT_API is used.',
        choices=['PySide6', 'PyQt6', 'PyQt5']
    )
    parser.add_argument(
        '--theme',
        help='The theme of the app, '
        + 'if not specified, the environment variable MITHEME is used.',
        choices=['None', 'qdarktheme', 'qdarkstyle', '...']
    )

    args, _ = parser.parse_known_args(sys.argv)

    # access the parsed arguments
    if args.QT_API:
        os.environ['QT_API'] = args.QT_API
        os.environ['PYQTGRAPH_QT_LIB'] = args.QT_API

    if args.theme:
        os.environ['MITHEME'] = args.theme

    return args


ARGS = getArgs()
