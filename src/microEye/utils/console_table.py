'''
Print simple tables to the console with rich (used instead of tabulate).

rich switches to plain ASCII borders when the console cannot show box-drawing
characters (e.g. a cp1252 Windows console), so printing never raises there.
Cells are shown as plain text: brackets in values such as '[1, 2]' are not
read as rich markup.

Keep tables out of log messages and exception texts: those should be one line.
'''

from collections.abc import Iterable, Sequence
from typing import Any, Optional

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text


def make_table(
    rows: Iterable[Sequence[Any]],
    headers: Sequence[Any],
    title: Optional[str] = None,
) -> Table:
    '''A rich Table of `rows` (each cell shown with str()).'''
    table = Table(
        title=None if title is None else Text(str(title)),
        box=box.ROUNDED,
    )
    for header in headers:
        table.add_column(Text(str(header)))
    for row in rows:
        table.add_row(*(Text(str(cell)) for cell in row))
    return table


def print_table(
    rows: Iterable[Sequence[Any]],
    headers: Sequence[Any],
    title: Optional[str] = None,
    console: Optional[Console] = None,
) -> None:
    '''Print a table of `rows` to `console` (default: a console on stdout).'''
    (console or Console()).print(make_table(rows, headers, title))
