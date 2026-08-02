from __future__ import annotations

import subprocess
import sys


def test_core_import_does_not_load_gui_or_experiment_packages() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, salvi; "
                "assert 'salvi.gui' not in sys.modules; "
                "assert 'salvi_experiments' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
