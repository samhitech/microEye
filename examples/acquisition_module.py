from microEye.hardware import acquisition_module

try:
    import vimba as vb
except Exception:
    vb = None

with vb.Vimba.get_instance() as vimba:
    app, window = acquisition_module.StartGUI()

    app.exec_()
