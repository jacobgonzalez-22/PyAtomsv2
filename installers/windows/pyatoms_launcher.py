"""
PyAtoms Standalone Launcher
Created on Wed Sep 02 16:54:20 PM
@author: Jacob

Modification Log
----------------
2026-09-03 - Jacob Gonzalez
    - Added null stdout/stderr handling for pyinstaller windowed builds

2026-09-02 - Jacob Gonzalez
    - Added launcher entry point for standalone PyInstaller builds
"""

import os
import sys

# PyInstaller windowed builds do not provide stdout/stderr.

# so redirect them to the null device so libraries that attempt
# console output do not crash the standalone application

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")

if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

from pyatoms.PyAtoms_GUI import main

if __name__ == "__main__":
    main()