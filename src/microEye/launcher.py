from microEye import ARGS, __version__


def splash(module):
    import random

    import pyfiglet
    import pyjokes

    fonts = [
        # 'ansi_shadow',
        # 'big',
        # 'doom',
        # 'slant',
        'small',
        'small_slant',
        'smslant',
        # 'standard',
        # 'stop',
    ]
    print(
        pyfiglet.Figlet(font=random.choice(fonts), width=250).renderText(
            f'MicroEye v{__version__}\n{module}'
        )
    )
    print(pyjokes.get_joke())


def main():
    # access the parsed arguments
    module_name = ARGS.module

    # import faulthandler
    # faulthandler.enable()

    gui_class = None
    name = {'mieye': 'miEye_module', 'viewer': 'multi_viewer'}.get(
        module_name, 'microLauncher'
    )

    splash(name)

    if module_name == 'mieye':
        from microEye.hardware.mieye import miEye_module

        gui_class = miEye_module
    elif module_name == 'viewer':
        from microEye.analysis.multi_viewer import multi_viewer

        gui_class = multi_viewer
    else:
        print('For help: \n>>> microEye -h')
        from microEye.utils.micro_launcher import microLauncher

        gui_class = microLauncher

    try:
        app, window = gui_class.StartGUI()
        app.exec()
    except Exception:
        import traceback

        traceback.print_exc()


if __name__ == '__main__':
    main()
