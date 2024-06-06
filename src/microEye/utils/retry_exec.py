import sys
import traceback

from microEye.qt import QApplication, QtWidgets


def retry_exec(func, *args, **kwargs):
    while True:
        try:
            func(*args, **kwargs)
            break
        except Exception as e:
            if not show_error_dialog(str(e)):
                break


def show_error_dialog(error_message):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    msg_box = QtWidgets.QMessageBox()
    msg_box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
    msg_box.setWindowTitle('Error')
    msg_box.setText('An error occurred')
    msg_box.setInformativeText(error_message)
    msg_box.setDetailedText(traceback.format_exc())
    retry_button = msg_box.addButton(
        'Retry', QtWidgets.QMessageBox.ButtonRole.AcceptRole
    )
    ok_button = msg_box.addButton(QtWidgets.QMessageBox.StandardButton.Ok)

    msg_box.exec()

    return msg_box.clickedButton() == retry_button
