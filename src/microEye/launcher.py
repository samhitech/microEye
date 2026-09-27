from rich.console import Console

from microEye import ARGS, HELP_MESSAGE, __version__


def splash(module):
    import random

    import pyfiglet
    import pyjokes
    from rich.panel import Panel
    from rich.text import Text

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
    title_panel = Panel(
        Text(
            pyfiglet.Figlet(font=random.choice(fonts), width=250).renderText(
                f'MicroEye v{__version__}'
            ),
            style='bold #dcdcaa',
        ),
        subtitle=f'[bold #4fc1ff]{module}[/bold #4fc1ff]',
        border_style='#4fc1ff',  # Bright VS Code blue for the outline
        padding=(1, 2),
        expand=False,
    )
    joke_text = Text.from_markup(
        f'\n[bold #dcdcaa]🤖 Random Joke:[/bold #dcdcaa] '
        f'[italic #ce9178]"{pyjokes.get_joke()}"[/italic #ce9178]\n'
    )

    console = Console()
    console.print(title_panel)
    console.print(joke_text)


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

    if ARGS.help:
        Console().print(HELP_MESSAGE)
        return

    if module_name == 'mieye':
        from microEye.hardware.mieye import miEye_module

        gui_class = miEye_module
    elif module_name == 'viewer':
        from microEye.analysis.multi_viewer import multi_viewer

        gui_class = multi_viewer
    else:
        Console().print(
            '[bold #dcdcaa]For help:\n>>>[/bold #dcdcaa] '
            '[italic #4fc1ff]microEye -h[/italic #4fc1ff]'
        )
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
