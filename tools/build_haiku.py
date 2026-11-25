"""Build the Haiku release zip.

    python3.14 tools/build_haiku.py [--out DIR]

PyInstaller has no Haiku bootloader, so on Haiku the add-on is not frozen: it
runs on HaikuPorts' Python. The zip holds, under the folder the manifest's
`executable` names:

    assemblyai-addon/assemblyai-addon   shell launcher (the manifest's executable)
    assemblyai-addon/boot.py            puts app/ and lib/ on sys.path, calls main()
    assemblyai-addon/app/               this add-on's own package
    assemblyai-addon/lib/               pure-Python wheels from haiku/requirements.txt

plus manifest.json, LICENSE and README.md at the root, as on the other
platforms. Everything else the add-on imports comes from Haiku packages the
manifest lists in `haiku_packages`; Subtitld checks they are installed and
offers to install the missing ones with pkgman.

Run it on Haiku when the add-on has native parts. This one is pure Python and
builds anywhere, which is how the release workflow builds it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import stat
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLATFORM_LABEL = 'haiku-x86_64'

LAUNCHER = '''#!/bin/sh
# Subtitld add-on launcher for Haiku: runs the add-on on HaikuPorts' Python.
here=$(cd "$(dirname "$0")" && pwd -P)
exec {python} -I "$here/boot.py" "$@"
'''

# -I: no PYTHONPATH, user site or script folder from whoever started us
# (Subtitld's own environment must not leak into the add-on).
BOOT = '''import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, 'app'), os.path.join(here, 'lib')]
sys.argv[0] = os.path.join(here, {exe_name!r})

from {module} import {function}

sys.exit({function}())
'''


def _entry_point(pyproject: dict, exe_name: str) -> tuple[str, str]:
    """`module`, `function` of the console script named like the executable."""
    scripts = pyproject.get('project', {}).get('scripts', {})
    target = scripts.get(exe_name)
    if not target:
        raise SystemExit(f'FATAL: pyproject.toml has no [project.scripts] entry {exe_name!r}')
    module, _, function = target.partition(':')
    return module, function or 'main'


def _python_for(manifest: dict) -> str:
    """The HaikuPorts Python the add-on runs on, from `haiku_packages`."""
    for name in manifest.get('haiku_packages') or []:
        if name.startswith('python3.') and '_' not in name:
            return name
    raise SystemExit('FATAL: manifest.json haiku_packages names no python3.x package')


def _write(path: Path, text: str, executable: bool = False) -> None:
    path.write_text(text, encoding='utf-8', newline='\n')
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _zip(folder: Path, archive: Path) -> None:
    """Zip `folder`'s contents, keeping Unix modes (the launcher must stay
    executable; Subtitld puts the bits back on extraction)."""
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                zf.write(path, path.relative_to(folder).as_posix())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', type=Path, default=ROOT, help='where the zip goes (default: repo root)')
    args = parser.parse_args(argv)

    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    pyproject = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    if 'haiku' not in (manifest.get('platforms') or []):
        raise SystemExit('FATAL: manifest.json platforms has no "haiku"')

    exe_rel = Path(manifest['executable'])
    exe_name = exe_rel.name
    module, function = _entry_point(pyproject, exe_name)
    packages = pyproject.get('tool', {}).get('setuptools', {}).get('packages') or [module.split('.')[0]]

    staging = ROOT / 'release-staging-haiku'
    shutil.rmtree(staging, ignore_errors=True)
    bundle = staging / exe_rel.parent
    (bundle / 'app').mkdir(parents=True)
    for package in packages:
        shutil.copytree(ROOT / package, bundle / 'app' / package,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))

    requirements = ROOT / 'haiku' / 'requirements.txt'
    (bundle / 'lib').mkdir()
    if requirements.is_file():
        # Pure-Python wheels only: anything with native code must be a Haiku
        # package, built for Haiku, not a wheel built for another system.
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-deps',
                        '--only-binary', ':all:', '--platform', 'any', '--implementation', 'py',
                        '--target', str(bundle / 'lib'), '-r', str(requirements)], check=True)

    _write(bundle / exe_name, LAUNCHER.format(python=_python_for(manifest)), executable=True)
    _write(bundle / 'boot.py', BOOT.format(exe_name=exe_name, module=module, function=function))

    for name in ('manifest.json', 'LICENSE', 'README.md'):
        shutil.copy2(ROOT / name, staging / name)
    if (ROOT / 'THIRD_PARTY_LICENSES').is_dir():
        shutil.copytree(ROOT / 'THIRD_PARTY_LICENSES', staging / 'THIRD_PARTY_LICENSES')

    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.out / f'{manifest["id"]}-{manifest["version"]}-{PLATFORM_LABEL}.zip'
    archive.unlink(missing_ok=True)
    _zip(staging, archive)
    print(archive)
    return 0


if __name__ == '__main__':
    sys.exit(main())
