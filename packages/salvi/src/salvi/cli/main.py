"""SALVI command-line entry point."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from salvi.cli.commands import dispatch
from salvi.cli.parser import build_parser
from salvi.exceptions import SalviError


def main(arguments: Sequence[str] | None = None) -> int:
    parser = build_parser()
    namespace = parser.parse_args(arguments)
    try:
        return dispatch(namespace)
    except SalviError as error:
        print(f"salvi: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
