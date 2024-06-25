import argparse
import os

from microEye import ARGS


def main():
    # access the parsed arguments
    module_name = ARGS.module

    if module_name == 'mieye':
        from microEye.hardware import miEye_module

        try:
            import vimba as vb
        except Exception:
            vb = None

        if vb:
            with vb.Vimba.get_instance() as vimba:
                app, window = miEye_module.StartGUI()
                app.exec()
        else:
            app, window = miEye_module.StartGUI()
            app.exec()
    elif module_name == 'viewer':
        from microEye.analysis.multi_viewer import multi_viewer

        app, window = multi_viewer.StartGUI('')

        app.exec()
    else:
        print('For help: \n>>> microEye -h')
        from microEye.utils.micro_launcher import microLauncher

        app, window = microLauncher.StartGUI()

        app.exec()


if __name__ == '__main__':
    main()
