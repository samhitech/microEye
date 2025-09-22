from microEye import ARGS


def main():
    # access the parsed arguments
    module_name = ARGS.module

    if module_name == 'mieye':
        # import faulthandler
        # faulthandler.enable()

        from microEye.hardware.mieye import miEye_module

        try:
            app, window = miEye_module.StartGUI()
            app.exec()
        except Exception:
            import traceback

            traceback.print_exc()
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
