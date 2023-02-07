from PyQt5.QtCore import QThread

panel = self.vimba_panels[0]

iterations = 5  # number of iterations
delay = 20  # delay in seconds
zstack = self.scanAcqWidget.z_acquire_btn  # the z-stack acquire btn


def loop(btn, iterations, delay):
    for it in range(iterations):
        btn.clicked.emit()
        QThread.msleep(delay*1000)
    
worker = thread_worker(
    loop, zstack, iterations, delay,
    progress=False, z_stage=False)
# Execute
worker.setAutoDelete(True)
self._threadpool.start(worker)
