"""Smoke-test a frozen build the way a user's machine runs it.

    python tools/smoke_frozen.py dist/assemblyai-addon/assemblyai-addon

Two checks against the real binary, over the real protocol:

  * **TLS.** SSL_CERT_FILE and SSL_CERT_DIR point at paths that do not
    exist, so the bundled OpenSSL finds no system CA store — what happens on
    macOS, Fedora/Arch and the Flatpak runtime, where the build machine's CA
    paths are absent. Without this, the CI runner's own CA store would hide
    the problem. The add-on must still verify Subtitld Cloud's certificate.
    The request carries a deliberately invalid key, so any HTTP answer (401,
    normally) proves the handshake worked.
  * **ffmpeg environment (Linux).** The frozen bootloader points
    LD_LIBRARY_PATH at the bundle's `_internal`; a system ffmpeg must not
    inherit that. A stand-in ffmpeg records what it was given.

POSIX only: the stand-in is a shell script, and on Windows Python reads the
system certificate store directly, so the env vars can't simulate the
failure there anyway.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

REQ_ID = 'smoke'

# Records LD_LIBRARY_PATH (or its absence), then writes the output file the
# add-on expects — the last argument, as with the real ffmpeg.
_FAKE_FFMPEG = '''#!/bin/sh
if [ "${LD_LIBRARY_PATH+set}" = set ]; then
    printf 'set:%s' "$LD_LIBRARY_PATH" > "$SMOKE_FFMPEG_REPORT"
else
    printf 'unset' > "$SMOKE_FFMPEG_REPORT"
fi
for last; do :; done
printf 'OggS smoke' > "$last"
'''


class Addon:
    def __init__(self, binary: Path, env: dict[str, str]):
        self.proc = subprocess.Popen(
            [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env)
        self.frames: queue.Queue = queue.Queue()
        self.stderr: list[str] = []
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stdout(self):
        for line in self.proc.stdout:
            self.frames.put(json.loads(line))
        self.frames.put(None)

    def _read_stderr(self):
        for line in self.proc.stderr:
            self.stderr.append(line.decode('utf-8', errors='replace').rstrip())

    def send(self, frame: dict) -> None:
        self.proc.stdin.write((json.dumps(frame) + '\n').encode())
        self.proc.stdin.flush()

    def next_frame(self, timeout: float) -> dict:
        try:
            frame = self.frames.get(timeout=timeout)
        except queue.Empty:
            self.fail(f'no frame within {timeout:.0f}s')
        if frame is None:
            self.fail('add-on exited early')
        return frame

    def transcribe(self, params: dict, timeout: float = 180.0) -> dict:
        """Handshake, run one request, return its terminal frame."""
        hello = self.next_frame(60)
        if hello.get('type') != 'hello':
            self.fail(f'expected hello, got {hello}')
        self.send({'type': 'ready', 'host': 'smoke', 'host_version': '0'})
        self.send({'type': 'asr.transcribe', 'id': REQ_ID, 'params': params})
        while True:
            frame = self.next_frame(timeout)
            if frame.get('id') == REQ_ID and frame.get('type') in ('error', 'result'):
                return frame

    def close(self) -> None:
        try:
            self.send({'type': 'shutdown'})
            self.proc.wait(timeout=15)
        except Exception:
            self.proc.kill()

    def fail(self, why: str):
        self.proc.kill()
        tail = '\n'.join(self.stderr[-30:])
        raise SystemExit(f'FAIL: {why}\n--- add-on stderr ---\n{tail}')


def run(binary: Path, work: Path, *, host_ld_path: str | None, base_url: str) -> tuple[dict, str]:
    """One add-on process, one request. Returns (terminal frame, ffmpeg report)."""
    report = work / 'ffmpeg-report.txt'
    report.unlink(missing_ok=True)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('ASSEMBLYAI_', 'SUBTITLD_CLOUD_', 'LD_LIBRARY_PATH'))}
    env.update({
        'SSL_CERT_FILE': str(work / 'no-such-ca-file.pem'),
        'SSL_CERT_DIR': str(work / 'no-such-ca-dir'),
        'SUBTITLD_FFMPEG_EXECUTABLE': str(work / 'ffmpeg'),
        'SMOKE_FFMPEG_REPORT': str(report),
    })
    if host_ld_path is not None:
        env['LD_LIBRARY_PATH'] = host_ld_path

    addon = Addon(binary, env)
    try:
        frame = addon.transcribe({
            'audio_path': str(work / 'in.wav'),
            'language': 'en',
            'options': {'api_key': 'smoke-test-invalid-key', 'base_url': base_url},
        })
    finally:
        addon.close()
    got = report.read_text() if report.exists() else '<ffmpeg never ran>'
    print(f'  terminal frame: {frame}')
    print(f'  ffmpeg saw LD_LIBRARY_PATH: {got}')
    return frame, got


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    if os.name != 'posix':
        print('skipped: POSIX only (see module docstring)')
        return 0
    binary = Path(sys.argv[1]).resolve()
    bundle_libs = str(binary.parent / '_internal')
    failures = []

    with tempfile.TemporaryDirectory(prefix='assemblyai-smoke-') as tmp:
        work = Path(tmp)
        (work / 'in.wav').write_bytes(b'RIFF smoke')
        fake = work / 'ffmpeg'
        fake.write_text(_FAKE_FFMPEG)
        fake.chmod(0o755)

        print('TLS with no system CA store (and LD_LIBRARY_PATH unset):')
        frame, seen = run(binary, work, host_ld_path=None,
                          base_url='https://cloud.subtitld.org')
        message = str(frame.get('message', ''))
        if frame.get('type') == 'error' and (
                'certificate' in message.lower() or message.startswith('Could not reach')):
            failures.append(f'TLS: the request never reached Subtitld Cloud: {message}')
        if sys.platform.startswith('linux') and seen != 'unset':
            failures.append(f'ffmpeg env: LD_LIBRARY_PATH should be unset, got {seen!r}')

        if sys.platform.startswith('linux'):
            host_libs = str(work / 'host-libs')
            print('ffmpeg environment with the user\'s own LD_LIBRARY_PATH:')
            # Nothing listens on the discard port: the request fails fast
            # once ffmpeg has run, which is all this run is for.
            _, seen = run(binary, work, host_ld_path=host_libs,
                          base_url='http://127.0.0.1:9')
            if seen != f'set:{host_libs}':
                failures.append(f'ffmpeg env: expected LD_LIBRARY_PATH={host_libs!r}, got {seen!r}')
            if bundle_libs in seen:
                failures.append(f'ffmpeg env: LD_LIBRARY_PATH still includes {bundle_libs}')

    for failure in failures:
        print(f'FAIL: {failure}')
    if not failures:
        print('OK')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
