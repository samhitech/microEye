import os


def check_modules(modules: list[str]) -> dict[str, bool]:
    availability = {}
    for module in modules:
        try:
            __import__(module)
            availability[module] = True
        except ImportError:
            availability[module] = False
    return availability

availability = check_modules(['PySide6', 'PyQt6', 'PyQt5'])

all_false = all(not value for value in availability.values())

if all_false:
    raise ImportError('Missing Qt packages, install one of PySide6, PyQt5, PyQt6.')

QT_API = os.environ.get('QT_API', 'PySide6')

if not availability.get(QT_API, False):
    for key, value in availability.items():
        if value:
            QT_API = key
            break


os.environ['QT_API'] = QT_API
os.environ['PYQTGRAPH_QT_LIB'] = QT_API

if QT_API == 'PySide6':
    from PySide6 import QtCore, QtGui, QtSerialPort, QtSvg, QtWidgets
    from PySide6.QtCore import QDateTime, Qt, Signal, Slot
    from PySide6.QtGui import QAction, QIcon
    from PySide6.QtWidgets import (
        QApplication,
        QFileSystemModel,
        QMainWindow,
        QVBoxLayout,
    )
elif QT_API == 'PyQt5':
    from PyQt5 import Qsci, QtCore, QtGui, QtSerialPort, QtSvg, QtWidgets
    from PyQt5.QtCore import QDateTime, Qt
    from PyQt5.QtCore import pyqtSignal as Signal
    from PyQt5.QtCore import pyqtSlot as Slot
    from PyQt5.QtGui import QIcon
    from PyQt5.QtWidgets import QAction, QApplication, QFileSystemModel, QMainWindow
elif QT_API == 'PyQt6':
    from PyQt6 import Qsci, QtCore, QtGui, QtSerialPort, QtSvg, QtWidgets
    from PyQt6.QtCore import QDateTime, Qt
    from PyQt6.QtCore import pyqtSignal as Signal
    from PyQt6.QtCore import pyqtSlot as Slot
    from PyQt6.QtGui import QAction, QFileSystemModel, QIcon
    from PyQt6.QtWidgets import QApplication, QMainWindow
else:
    raise ImportError('QT_API environment variable is not correctly set. \
                      It should be one of PySide6, PyQt5, PyQt6.')


def checkDialogOptions(options) -> QtWidgets.QFileDialog.Option:
    if options is None or not isinstance(options, QtWidgets.QFileDialog.Option):
        options = QtWidgets.QFileDialog.Option(0)
    return options


def getOpenFileName(
    parent: QtWidgets.QWidget = None,
    caption='Open File',
    directory='',
    filter='All Files (*)',
    initial_filter='',
    options: QtWidgets.QFileDialog.Option =None,
):
    options = checkDialogOptions(options)
    if QT_API == 'PySide6':
        return QtWidgets.QFileDialog.getOpenFileName(
            parent,
            caption,
            dir=directory,
            filter=filter,
            selectedFilter=initial_filter,
            options=options,
        )
    elif QT_API == 'PyQt6' or QT_API == 'PyQt5':
        return QtWidgets.QFileDialog.getOpenFileName(
            parent,
            caption,
            directory=directory,
            filter=filter,
            initialFilter=initial_filter,
            options=options,
        )


def getOpenFileNames(
    parent: QtWidgets.QWidget = None,
    caption='Save File',
    directory='',
    filter='All Files (*)',
    initial_filter='',
    options: QtWidgets.QFileDialog.Option =None,
):
    options = checkDialogOptions(options)
    if QT_API == 'PySide6':
        return QtWidgets.QFileDialog.getOpenFileNames(
            parent,
            caption,
            dir=directory,
            filter=filter,
            selectedFilter=initial_filter,
            options=options,
        )
    elif QT_API == 'PyQt6' or QT_API == 'PyQt5':
        return QtWidgets.QFileDialog.getOpenFileNames(
            parent,
            caption,
            directory=directory,
            filter=filter,
            initialFilter=initial_filter,
            options=options,
        )


def getSaveFileName(
    parent: QtWidgets.QWidget = None,
    caption='Save File',
    directory='',
    filter='All Files (*)',
    initial_filter='',
    options: QtWidgets.QFileDialog.Option =None,
):
    options = checkDialogOptions(options)
    if QT_API == 'PySide6':
        return QtWidgets.QFileDialog.getSaveFileName(
            parent,
            caption,
            dir=directory,
            filter=filter,
            selectedFilter=initial_filter,
            options=options,
        )
    elif QT_API == 'PyQt6' or QT_API == 'PyQt5':
        return QtWidgets.QFileDialog.getSaveFileName(
            parent,
            caption,
            directory=directory,
            filter=filter,
            initialFilter=initial_filter,
            options=options,
        )


def getExistingDirectory(
    parent: QtWidgets.QWidget = None,
    caption='Save File',
    directory='',
    filter='All Files (*)',
    initial_filter='',
    options: QtWidgets.QFileDialog.Option =None,
):
    options = checkDialogOptions(options)
    if QT_API == 'PySide6':
        return QtWidgets.QFileDialog.getExistingDirectory(
            parent,
            caption,
            dir=directory,
            filter=filter,
            selectedFilter=initial_filter,
            options=options,
        )
    elif QT_API == 'PyQt6' or QT_API == 'PyQt5':
        return QtWidgets.QFileDialog.getExistingDirectory(
            parent,
            caption,
            directory=directory,
            filter=filter,
            initialFilter=initial_filter,
            options=options,
        )
