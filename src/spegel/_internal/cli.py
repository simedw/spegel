# Why does this file exist, and why not put this in `__main__`?
#
# You might be tempted to import things from `__main__` later,
# but that will cause problems: the code will get executed twice:
#
# - When you run `python -m bear_utils` python will execute
#   `__main__.py` as a script. That means there won't be any
#   `bear_utils.__main__` in `sys.modules`.
# - When you import `__main__` it will get executed again (as a module) because
#   there's no `bear_utils.__main__` in `sys.modules`.
from __future__ import annotations

from argparse import Action, ArgumentParser, Namespace
import sys
from typing import Any

from spegel._internal import debug
from spegel.main import Spegel

class _DebugInfo(Action):
    def __init__(self, nargs: int | str | None = 0, **kwargs: Any) -> None:
        super().__init__(nargs=nargs, **kwargs)

    def __call__(self, *_: Any, **__: Any) -> None:
        print(debug._get_debug_info())
        sys.exit(0)


class _About(Action):
    def __init__(self, nargs: int | str | None = 0, **kwargs: Any) -> None:
        super().__init__(nargs=nargs, **kwargs)

    def __call__(self, *_: Any, **__: Any) -> None:
        print(debug._get_package_info())
        sys.exit(0)


def get_parser() -> ArgumentParser:
    name: str = debug._get_name()
    version: str = f"{name} v{debug._get_version()}"
    parser = ArgumentParser(description=name.capitalize(), prog=name, exit_on_error=False)
    parser.add_argument("-V", "--version", action="version", version=version)
    parser.add_argument("--about", action=_About, help="Print information about the package")
    parser.add_argument("--debug_info", action=_DebugInfo, help="Print debug information")
    parser.add_argument("url", nargs="?", help="URL to open immediately on launch")
    return parser


def main(args: list[str] | None = None) -> int:
    """Main entry point for the CLI.

    This function is called when the CLI is executed. It can be used to
    initialize the CLI, parse arguments, and execute commands.

    Args:
        args (list[str] | None): A list of command-line arguments. If None, uses sys.argv[1:].

    Returns:
        int: Exit code of the CLI execution. 0 for success, non-zero for failure.
    """
    if args is None:
        args = sys.argv[1:]
    try:
        parser: ArgumentParser = get_parser()
        opts: Namespace = parser.parse_args(args)
        initial_url: str | None = opts.url
        
        if initial_url is not None and not initial_url.startswith(("http://", "https://")):
            # Auto-prepend https if scheme is missing
            initial_url = f"https://{initial_url}"

        app = Spegel(initial_url=initial_url)
        app.run()
    except Exception as e:
        print(f"Error initializing {debug._get_name()}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    main()
