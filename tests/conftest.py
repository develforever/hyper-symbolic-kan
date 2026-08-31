"""Pytest bootstrap: make the repository root importable as ``src.*`` / ``tests.*``.

Shared native-extension markers live in :mod:`tests._native`.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
