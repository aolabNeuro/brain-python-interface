# -*- mode: python ; coding: utf-8 -*-

import sys

sys.setrecursionlimit(sys.getrecursionlimit() * 5)


a = Analysis(
    ['demo_tasks/demo_tracking_task.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('features/images', 'features/images'),
        ('riglib/audio', 'riglib/audio'),
        ('riglib/stereo_opengl/shaders/*.glsl', 'riglib/stereo_opengl/shaders'),
        ('db/db_test_aopy.sql', 'db'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'pytest',
        'sphinx',
        'numba',
        'llvmlite',
        'dask',
        'pyarrow',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='demo-tracking-launcher',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
app = BUNDLE(
    exe,
    name='demo-tracking-launcher.app',
    icon=None,
    bundle_identifier=None,
)
