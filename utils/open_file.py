"""
utils/open_file.py
==================
Tiny helper to open a saved file in the operating system's default viewer.

Used by the plotting tools so that, after a figure is saved, the result pops
open automatically instead of having to be found in outputs/ by hand.

Opening is always best-effort: any failure is caught and logged, never raised.
The file has already been saved by the time this is called -- opening it is
purely a convenience.
"""
import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def open_in_viewer(path: Path) -> None:
    """
    Open a file in the OS default application, best-effort.

    Windows uses os.startfile; macOS uses `open`; Linux uses `xdg-open`. Any
    error (no viewer, headless session, missing file) is caught and logged --
    it never interrupts the caller.

    Args:
        path: Path to the file to open.
    """
    path = Path(path)
    if not path.exists():
        log.warning("cannot open %s -- file does not exist", path)
        return

    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))                       # noqa: S606 (Windows)
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
        log.info("opened %s", path)
    except Exception as exc:                              # best-effort only
        log.warning("could not open %s in a viewer: %s", path, exc)
