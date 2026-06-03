from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class WindowBounds:
    x: int
    y: int
    width: int
    height: int


def primary_work_area() -> WindowBounds:
    """Return the primary monitor work area (excludes taskbar) on Windows."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", wintypes.LONG),
                ("top", wintypes.LONG),
                ("right", wintypes.LONG),
                ("bottom", wintypes.LONG),
            ]

        rect = RECT()
        # SPI_GETWORKAREA = 0x0030
        if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
            return WindowBounds(
                x=rect.left,
                y=rect.top,
                width=max(rect.right - rect.left, 640),
                height=max(rect.bottom - rect.top, 480),
            )

    return WindowBounds(x=0, y=0, width=1040, height=760)
