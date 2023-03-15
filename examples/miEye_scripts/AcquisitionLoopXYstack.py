from PyQt5.QtCore import QThread

iterations = 5  # number of iterations
delay = 10  # delay in seconds


def loop(mieye, iterations, delay):
    for it in range(iterations):
        if not mieye._scanning:
            mieye._stop_scan = False
            mieye._scanning = True

            params = mieye.scanAcqWidget.get_params()

            mieye.scan_worker = thread_worker(
                scan_acquisition, mieye,
                [params[0], params[1]],
                [params[2], params[3]],
                params[4],
                params[5], progress=False, z_stage=False)
            mieye.scan_worker.signals.result.connect(
                mieye.result_scan_export)
            # Execute
            mieye._threadpool.start(mieye.scan_worker)

            mieye.scanAcqWidget.acquire_btn.setEnabled(False)
            mieye.scanAcqWidget.z_acquire_btn.setEnabled(False)
        QThread.msleep(delay*1000)


worker = thread_worker(
    loop, self, iterations, delay,
    progress=False, z_stage=False)
# Execute
worker.setAutoDelete(True)
self._threadpool.start(worker)
