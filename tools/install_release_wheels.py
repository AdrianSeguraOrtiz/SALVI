"""Install the two publishable SALVI wheels into the active clean environment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    wheels = {
        path.name.split("-", maxsplit=1)[0]: path for path in arguments.directory.glob("*.whl")
    }
    required = {"salvi", "salvi_experiments"}
    if set(wheels) != required:
        parser.error(f"expected exactly {sorted(required)!r} wheels, found {sorted(wheels)!r}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            str(wheels["salvi"]),
            str(wheels["salvi_experiments"]),
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
