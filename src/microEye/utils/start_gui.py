import os
import random
import sys

import pyfiglet
import pyjokes
import qdarkstyle
import qdarktheme

from microEye.qt import QT_API, QApplication, QtCore, QtGui, QtWidgets


def splash(module):
    fonts = [
        # 'ansi_shadow',
        # 'big',
        # 'doom',
        # 'slant',
        'small',
        'small_slant',
        'smslant',
        # 'standard',
        # 'stop',
    ]
    print(
        pyfiglet.Figlet(font=random.choice(fonts), width=250).renderText(
            f'MicroEye v2.1.0\n{module}'
        )
    )


QSS = '''
    ParameterControlledButton {
        font-weight: 600;
        min-width: 100px;
        min-height: 20px;
    }
    QPushButton {
        padding: 4px;
        font-weight: 600;
    }
    QDockWidget {
        font-weight: 600;
        font-size: 11pt;
    }
    microLauncher {
        border: 1px solid white;
        padding: 0;
    }
    QPushButton#closeButton {
        border: none;
    }
    '''


def StartGUI(cls: type, *args, **kwargs):
    '''Initializes a new QApplication and module.

    Use
    -------
    app, window = StartGUI(module_type)


    app.exec()

    Returns
    -------
    tuple (QApplication, module)
        Returns a tuple with QApp and module window.
    '''
    splash(cls.__name__)
    print(pyjokes.get_joke())
    # create a QApp
    app = QApplication(sys.argv)
    # set darkmode from *qdarkstyle* (not compatible with pyqt6)
    # Additional stylesheet

    theme = os.environ.get('MITHEME', default=None)
    if theme is not None:
        if theme == 'qdarkstyle':
            app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api=QT_API))
        elif theme == 'qdarktheme':
            qdarktheme.setup_theme(additional_qss=QSS)
        elif theme in QtWidgets.QStyleFactory.keys():  # noqa: SIM118
            app.setStyle(theme)

    # sets the app icon
    dirname = os.path.dirname(os.path.abspath(__file__))
    app_icon = QtGui.QIcon()
    app_icon.addFile(os.path.join(dirname, '../icons/16.png'), QtCore.QSize(16, 16))
    app_icon.addFile(os.path.join(dirname, '../icons/24.png'), QtCore.QSize(24, 24))
    app_icon.addFile(os.path.join(dirname, '../icons/32.png'), QtCore.QSize(32, 32))
    app_icon.addFile(os.path.join(dirname, '../icons/48.png'), QtCore.QSize(48, 48))
    app_icon.addFile(os.path.join(dirname, '../icons/64.png'), QtCore.QSize(64, 64))
    app_icon.addFile(os.path.join(dirname, '../icons/128.png'), QtCore.QSize(128, 128))
    app_icon.addFile(os.path.join(dirname, '../icons/256.png'), QtCore.QSize(256, 256))
    app_icon.addFile(os.path.join(dirname, '../icons/512.png'), QtCore.QSize(512, 512))

    app.setWindowIcon(app_icon)

    if sys.platform.startswith('win'):
        import ctypes

        myappid = f'samhitech.mircoEye.{cls.__name__}'  # appid
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    window = cls(*args)
    return app, window
