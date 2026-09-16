# SPDX-FileCopyrightText: 2024-present barrettMCW <mjbarrett@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""SHELL — SHELL Highlights Epithelium and Lumen Locations.

This module intentionally keeps the top-level package import lightweight.
Only the package version is exported here so importing ``shell`` in tests
or simple checks does not eagerly import heavy runtime dependencies
(like ``torch``, ``monai``, ``pyvips``, or ``omero``).
"""

import warnings

from .__about__ import __version__

warnings.filterwarnings(
    "ignore",
    message=r".*torch.*jit.*interface.*",
    category=FutureWarning,
)

__all__ = ["__version__"]
