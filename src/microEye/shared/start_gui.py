import os
import random
import sys

import pyfiglet
import qdarkstyle
import qdarktheme
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtSerialPort import *
from PyQt5.QtWidgets import *


def splash(module):
    fonts = [
        'ansi_shadow', 'big', 'doom', 'slant',
        'small', 'small_slant', 'smslant', 'standard', 'stop']
    print(
        pyfiglet.Figlet(
            font=random.choice(fonts), width=250).renderText(
                 f'MicroEye v2.0.0\n{module}'))

def StartGUI(cls: type, *args, **kwargs):
        '''Initializes a new QApplication and module.

        Use
        -------
        app, window = StartGUI(module_type)


        app.exec_()

        Returns
        -------
        tuple (QApplication, module)
            Returns a tuple with QApp and module window.
        '''
        splash(cls.__name__)
        # create a QApp
        app = QApplication(sys.argv)
        # set darkmode from *qdarkstyle* (not compatible with pyqt6)
        # Additional stylesheet
        qss = '''
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
        '''
        if 'theme' in kwargs:
            theme = kwargs.get('theme', 'qdarkstyle')
            if theme == 'qdarkstyle':
                app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
            else:
                qdarktheme.setup_theme(additional_qss=qss)
        else:
            qdarktheme.setup_theme(additional_qss=qss)
        # sets the app icon
        dirname = os.path.dirname(os.path.abspath(__file__))
        app_icon = QIcon()
        app_icon.addFile(
            os.path.join(dirname, '../icons/16.png'), QSize(16, 16))
        app_icon.addFile(
            os.path.join(dirname, '../icons/24.png'), QSize(24, 24))
        app_icon.addFile(
            os.path.join(dirname, '../icons/32.png'), QSize(32, 32))
        app_icon.addFile(
            os.path.join(dirname, '../icons/48.png'), QSize(48, 48))
        app_icon.addFile(
            os.path.join(dirname, '../icons/64.png'), QSize(64, 64))
        app_icon.addFile(
            os.path.join(dirname, '../icons/128.png'), QSize(128, 128))
        app_icon.addFile(
            os.path.join(dirname, '../icons/256.png'), QSize(256, 256))
        app_icon.addFile(
            os.path.join(dirname, '../icons/512.png'), QSize(512, 512))

        app.setWindowIcon(app_icon)

        if sys.platform.startswith('win'):
            import ctypes
            myappid = f'samhitech.mircoEye.{cls.__name__}'  # appid
            ctypes.windll.shell32.\
                SetCurrentProcessExplicitAppUserModelID(myappid)

        window = cls(*args)
        return app, window
