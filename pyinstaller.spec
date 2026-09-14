# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Subtitld AssemblyAI add-on.

Produces `dist/assemblyai-addon/` containing the launcher binary
(`assemblyai-addon`, `.exe` on Windows) and nothing else of substance — the
add-on is stdlib-only, so the bundle is essentially CPython plus one module.

Run from the addon root:

    pyinstaller pyinstaller.spec --noconfirm

The release workflow zips `dist/assemblyai-addon/` together with
`manifest.json`, `LICENSE`, and `README.md` into the platform-tagged archive
consumed by Subtitld's AddonsDialog.
"""

# ruff: noqa: F821  # PyInstaller injects Analysis/PYZ/EXE/COLLECT at runtime.

from __future__ import annotations

import os
from pathlib import Path

SPEC_ROOT = Path(SPECPATH).resolve()

a = Analysis(
    [str(SPEC_ROOT / 'assemblyai_addon' / '__main__.py')],
    pathex=[str(SPEC_ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # No Qt, no Tk, no scientific stack — this add-on is an HTTP client.
        # Excluding them keeps the bundle around 10 MB instead of 80+.
        'PySide6', 'PyQt6', 'PyQt5',
        'tkinter', 'Tkinter', '_tkinter',
        'matplotlib', 'numpy', 'scipy', 'pandas',
        'sqlite3', 'unittest', 'pydoc_data',
    ],
    noarchive=False,
)


# Strip the bundled libexpat. PyInstaller copies whatever expat is on the
# build host; on modern Ubuntu (22.04+) `libpython3.12` links against expat
# >= 2.6, which exposes `XML_SetReparseDeferralEnabled`. Many user systems
# have an older expat, so the bundled .so lacks that symbol and libpython
# fails to load with
#
#     undefined symbol: XML_SetReparseDeferralEnabled
#
# Dropping our copy lets the dynamic linker pick the system one, which every
# distro ships. On macOS / Windows the binaries list has no libexpat, so this
# is a no-op there.
def _drop_bundled_libexpat(entries):
    keep = []
    for entry in entries:
        # Each entry is (dest_name, src_path, type_code).
        name = entry[0] if entry else ''
        base = os.path.basename(name)
        if base.startswith('libexpat.so') or base.startswith('libexpat.dylib'):
            continue
        keep.append(entry)
    return keep


a.binaries = _drop_bundled_libexpat(a.binaries)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='assemblyai-addon',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # Communication is over stdio — no GUI window.
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='assemblyai-addon',
)
