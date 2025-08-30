# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['veritas_final_ui.py'],
    pathex=[],
    binaries=[],
    datas=[('background.png', '.'), ('splash.png', '.'), ('siren.wav', '.'), ('yolov8n.pt', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='veritas_final_ui',
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
    icon=['loco.ico'],
)
