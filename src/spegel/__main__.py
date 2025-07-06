"""Run Spegel as a module: `python -m spegel`"""
import sys

from ._internal.cli import main as cli_main

if __name__ == "__main__":
    sys.exit(cli_main(sys.argv[1:]))  # Pass command-line arguments to the CLI main function
