"""Availability of the compiled ``_cpp_kernels`` extension, for test gating.

Audit V2: the native tests used a bare ``assert engine.is_native_available()``,
so the whole suite failed on a clean checkout without a compiled extension.
They are now skipped when ``_cpp_kernels`` is missing -- *except* when
``HSKAN_REQUIRE_NATIVE=1`` is set, which turns the skip into a hard failure.

The CI matrix runs the suite normally; a separate CI job builds the extension
and runs with ``HSKAN_REQUIRE_NATIVE=1`` (see ``.github/workflows/ci.yml``, job
``native-required``), so a skip that silently disables the entire native path
cannot pass unnoticed.
"""

import os

import pytest

from src.cpp_kernels.cpp_kan_engine import _HAS_CPP


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no")


REQUIRE_NATIVE = _env_flag("HSKAN_REQUIRE_NATIVE")

requires_native = pytest.mark.skipif(
    not _HAS_CPP and not REQUIRE_NATIVE,
    reason=(
        "native extension _cpp_kernels is not built; "
        "set HSKAN_REQUIRE_NATIVE=1 to fail instead of skip"
    ),
)
