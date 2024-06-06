from microEye.QtCore import QThread

panel = self.vimba_panels[0]

iterations = 5  # number of iterations
delay = 10  # delay in seconds
freerun = panel.cam_freerun_btn  # the panel acquire btn


def loop(freerun, iterations, delay):
    for it in range(iterations):
        freerun.clicked.emit()
        QThread.msleep(delay * 1000)


worker = thread_worker(loop, freerun, iterations, delay, progress=False, z_stage=False)
# Execute
worker.setAutoDelete(True)
self._threadpool.start(worker)
