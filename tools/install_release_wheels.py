"""Install the publishable SALVI wheel into the active clean environment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    wheels = tuple(arguments.directory.glob("salvi-*.whl"))
    if len(wheels) != 1:
        parser.error(f"expected exactly one SALVI wheel, found {len(wheels)}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            str(wheels[0]),
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
