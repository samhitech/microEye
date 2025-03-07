from microEye.hardware.cams import CameraList, Vimba_Panel
from microEye.qt import QtCore
from microEye.utils.thread_worker import QThreadWorker

panel: Vimba_Panel = CameraList.CAMERAS['Vimba'][0]

iterations = 5  # number of iterations
delay = 10  # delay in seconds
freerun = panel.camera_options.search_param('freerun')  # the panel acquire btn

if freerun:
    def loop(freerun, iterations, delay, **kwargs):
        for _ in range(iterations):
            freerun.sigActivated.emit()
            QtCore.QThread.msleep(500)
            panel.Event.wait()
            QtCore.QThread.msleep(delay * 1000)


    worker = QThreadWorker(loop, freerun, iterations, delay)
    # Execute
    worker.setAutoDelete(True)
    QtCore.QThreadPool.globalInstance().start(worker)
