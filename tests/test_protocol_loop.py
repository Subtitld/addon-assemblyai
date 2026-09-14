"""Tests for the AssemblyAI add-on.

Three layers:

  * **Pure translation** — the `sentences[]` / `utterances[]` / `words[]`
    decision table and the ms-vs-seconds sniffing. No I/O.
  * **Error mapping** — every documented cloud status maps to a code the
    host understands plus a message a user can act on.
  * **End-to-end** — the add-on driven as a real subprocess over stdio
    against a real (local, fake) cloud server. This exercises the actual
    urllib calls, the hand-rolled multipart encoder, the polling loop, and
    cancellation — the parts most likely to break silently.

The e2e tests need `ffmpeg` and are skipped without it, so the pure layers
still run on a bare CI image.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from assemblyai_addon import __main__ as addon  # noqa: E402

HAS_FFMPEG = shutil.which('ffmpeg') is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason='ffmpeg not installed')


# ---------------------------------------------------------------------------
# Translation: sentences / utterances / words / text
# ---------------------------------------------------------------------------
def test_sentences_preferred_over_utterances_and_words():
    """All three shapes present → sentences wins, verbatim."""
    job = {
        'audio_duration': 10.0,
        'sentences': [{'start': 0, 'end': 2000, 'text': 'From sentences.', 'speaker': 'A'}],
        'utterances': [{'start': 0, 'end': 9000, 'text': 'From utterances.', 'speaker': 'A'}],
        'words': [{'start': 0, 'end': 500, 'text': 'From'}],
    }
    segs = addon._segments_from_job(job)
    assert [s['text'] for s in segs] == ['From sentences.']


def test_utterances_used_when_no_sentences():
    job = {
        'audio_duration': 10.0,
        'utterances': [
            {'start': 0, 'end': 3000, 'text': 'Hello there.', 'speaker': 'A'},
            {'start': 3200, 'end': 6000, 'text': 'General Kenobi.', 'speaker': 'B'},
        ],
        'words': [{'start': 0, 'end': 500, 'text': 'Hello'}],
    }
    segs = addon._segments_from_job(job)
    assert [s['speaker'] for s in segs] == ['A', 'B']
    assert segs[0]['end'] == pytest.approx(3.0)


def test_words_fallback_groups_on_pause_and_speaker():
    job = {
        'audio_duration': 20.0,
        'words': [
            {'start': 0, 'end': 400, 'text': 'one', 'speaker': 'A'},
            {'start': 420, 'end': 800, 'text': 'two', 'speaker': 'A'},
            # 1.2 s gap — past _WORD_GAP_BREAK_SEC, so a new cue starts.
            {'start': 2000, 'end': 2400, 'text': 'three', 'speaker': 'A'},
            # Speaker change forces a break even without a gap.
            {'start': 2450, 'end': 2800, 'text': 'four', 'speaker': 'B'},
        ],
    }
    segs = addon._segments_from_job(job)
    assert [s['text'] for s in segs] == ['one two', 'three', 'four']
    assert [s['speaker'] for s in segs] == ['A', 'A', 'B']


def test_words_fallback_runaway_duration_cap():
    """A speaker who never pauses still gets broken up eventually."""
    words = []
    t = 0
    while t < 60_000:  # 60 s of back-to-back words, no gap
        words.append({'start': t, 'end': t + 300, 'text': 'word', 'speaker': 'A'})
        t += 310
    segs = addon._segments_from_job({'audio_duration': 62.0, 'words': words})
    assert len(segs) > 1
    for seg in segs:
        assert seg['end'] - seg['start'] <= addon._PHRASE_HARD_MAX_DURATION_SEC + 1.0


def test_bare_text_fallback_uses_local_duration():
    """No timing data at all → one whole-audio cue with a real end time.

    A zero-length cue would draw at zero width and divide-by-zero the
    speaker-percentage panel, so the locally-probed duration has to land.
    """
    segs = addon._segments_from_job(
        {'text': 'Just a flat transcript.'},
        fallback_duration_sec=42.5,
    )
    assert len(segs) == 1
    assert segs[0]['end'] == pytest.approx(42.5)


def test_empty_response_returns_no_segments():
    """Truly empty stays empty — the import view needs to say 'nothing found'
    rather than show a blank cue."""
    assert addon._segments_from_job({'audio_duration': 10.0}) == []
    assert addon._segments_from_job({'text': '   '}) == []


def test_slice_by_phrase_off_merges_same_speaker_runs():
    job = {
        'audio_duration': 12.0,
        'sentences': [
            {'start': 0, 'end': 2000, 'text': 'One.', 'speaker': 'A'},
            {'start': 2000, 'end': 4000, 'text': 'Two.', 'speaker': 'A'},
            {'start': 4000, 'end': 6000, 'text': 'Three.', 'speaker': 'B'},
        ],
    }
    sliced = addon._segments_from_job(job, slice_by_phrase=True)
    merged = addon._segments_from_job(job, slice_by_phrase=False)
    assert len(sliced) == 3
    assert [s['text'] for s in merged] == ['One. Two.', 'Three.']
    # The merged cue must span the whole run, not just the first sentence.
    assert merged[0]['end'] == pytest.approx(4.0)


def test_merge_keeps_undiarized_transcript_contiguous():
    """Empty speakers all compare equal, so an undiarized transcript
    collapses to one cue when slicing is off."""
    segs = [
        {'start': 0.0, 'end': 1.0, 'text': 'a', 'speaker': ''},
        {'start': 1.0, 'end': 2.0, 'text': 'b', 'speaker': ''},
    ]
    merged = addon._merge_consecutive_speakers(segs)
    assert len(merged) == 1
    assert merged[0]['text'] == 'a b'


# ---------------------------------------------------------------------------
# Unit sniffing
# ---------------------------------------------------------------------------
def test_ms_detected_against_known_duration():
    # 8000 in a 10 s file can only be milliseconds.
    assert addon._detect_ms_divisor(8000.0, 10.0) == 1000.0
    # 8 in a 10 s file is already seconds.
    assert addon._detect_ms_divisor(8.0, 10.0) == 1.0


def test_ms_detected_without_known_duration():
    """No duration → anything past 24 h of seconds must be ms."""
    assert addon._detect_ms_divisor(90_000.0, 0.0) == 1000.0
    assert addon._detect_ms_divisor(3600.0, 0.0) == 1.0


def test_seconds_payload_is_not_rescaled():
    """A cloud that already converted to seconds must pass through intact."""
    job = {
        'audio_duration': 10.0,
        'sentences': [{'start': 1.5, 'end': 4.25, 'text': 'Already seconds.'}],
    }
    seg = addon._segments_from_job(job)[0]
    assert seg['start'] == pytest.approx(1.5)
    assert seg['end'] == pytest.approx(4.25)


def test_ms_duration_field_is_normalised():
    """`audio_duration` itself may arrive in ms."""
    job = {'audio_duration': 120_000, 'text': 'hi'}
    assert addon._segments_from_job(job)[0]['end'] == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------
def test_model_id_expansion():
    assert addon._resolve_model({'model': 'nano'}) == 'subtitld-cloud:assemblyai/nano'
    assert addon._resolve_model({'model': 'assemblyai/best'}) == 'subtitld-cloud:assemblyai/best'
    full = 'subtitld-cloud:assemblyai/best'
    assert addon._resolve_model({'model': full}) == full
    assert addon._resolve_model({}) == full  # default tier


def test_config_precedence_options_over_env(monkeypatch):
    monkeypatch.setenv('ASSEMBLYAI_API_KEY', 'from-addon-config')
    monkeypatch.setenv('SUBTITLD_CLOUD_API_KEY', 'from-shared-slot')
    assert addon._resolve_api_key({'api_key': 'from-request'}) == 'from-request'
    assert addon._resolve_api_key({}) == 'from-addon-config'
    monkeypatch.delenv('ASSEMBLYAI_API_KEY')
    # Falls through to the shared cloud slot rather than demanding a
    # second copy of the same key.
    assert addon._resolve_api_key({}) == 'from-shared-slot'


def test_base_url_defaults_and_strips_slash(monkeypatch):
    monkeypatch.delenv('ASSEMBLYAI_BASE_URL', raising=False)
    monkeypatch.delenv('SUBTITLD_CLOUD_BASE_URL', raising=False)
    assert addon._resolve_base_url({}) == 'https://cloud.subtitld.org'
    assert addon._resolve_base_url({'base_url': 'https://x.test/'}) == 'https://x.test'
    monkeypatch.setenv('SUBTITLD_CLOUD_BASE_URL', addon.DRAFT_BASE_URL)
    assert addon._resolve_base_url({}) == addon.DRAFT_BASE_URL


def test_explicit_false_option_survives(monkeypatch):
    """`options.get(k) or default` would turn an explicit False back into
    the default — the exact bug this helper exists to avoid."""
    monkeypatch.delenv('ASSEMBLYAI_SPEAKER_LABELS', raising=False)
    assert addon._opt_bool({'speaker_labels': False}, 'speaker_labels',
                           'ASSEMBLYAI_SPEAKER_LABELS', True) is False
    assert addon._opt_bool({}, 'speaker_labels',
                           'ASSEMBLYAI_SPEAKER_LABELS', True) is True
    monkeypatch.setenv('ASSEMBLYAI_SPEAKER_LABELS', '0')
    assert addon._opt_bool({}, 'speaker_labels',
                           'ASSEMBLYAI_SPEAKER_LABELS', True) is False


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------
BASE = 'https://cloud.subtitld.org'


def test_401_names_the_key_and_is_not_retryable():
    code, msg, retry = addon._translate_http_error(
        addon.CloudHTTPError(401, '{}'), BASE, 'submit')
    assert code == addon.ERR_BAD_PARAMS
    assert 'Invalid API key' in msg
    assert retry is False


def test_402_points_at_topup():
    code, msg, retry = addon._translate_http_error(
        addon.CloudHTTPError(402, '{}'), BASE, 'submit')
    assert 'Insufficient balance' in msg
    assert f'{BASE}/dashboard/topup/' in msg
    assert retry is False


def test_404_on_submit_is_unknown_model_but_on_poll_is_not():
    """Same status, different cause — the message has to say which."""
    code, msg, _ = addon._translate_http_error(
        addon.CloudHTTPError(404, '{}'), BASE, 'submit')
    assert code == addon.ERR_MODEL_MISSING
    assert 'Unknown model' in msg

    code, msg, _ = addon._translate_http_error(
        addon.CloudHTTPError(404, '{}'), BASE, 'poll')
    assert 'job id' in msg


@pytest.mark.parametrize('kind,needle', [
    ('upload_failed', 'could not forward'),
    ('provider_not_configured', 'not configured on the Subtitld Cloud server'),
    ('upload_rejected_by_provider', 'rejected the uploaded audio'),
])
def test_503_typed_errors_get_specific_messages(kind, needle):
    exc = addon.CloudHTTPError(503, json.dumps({'error': kind}))
    code, msg, _ = addon._translate_http_error(exc, BASE, 'upload')
    assert code == addon.ERR_NETWORK_UNAVAILABLE
    assert needle in msg


def test_503_includes_upstream_status_when_present():
    exc = addon.CloudHTTPError(
        503, json.dumps({'error': 'upload_rejected_by_provider', 'upstream_status': 415}))
    _, msg, _ = addon._translate_http_error(exc, BASE, 'upload')
    assert 'upstream status 415' in msg


def test_provider_not_configured_is_not_retryable():
    """Retrying a server-side misconfiguration just hammers a broken
    endpoint — everything else transient stays retryable."""
    misconfig = addon.CloudHTTPError(503, json.dumps({'error': 'provider_not_configured'}))
    transient = addon.CloudHTTPError(503, json.dumps({'error': 'upload_failed'}))
    assert addon._translate_http_error(misconfig, BASE, 'upload')[2] is False
    assert addon._translate_http_error(transient, BASE, 'upload')[2] is True


def test_unknown_5xx_is_retryable():
    assert addon._translate_http_error(
        addon.CloudHTTPError(500, ''), BASE, 'poll')[2] is True


def test_cloud_error_survives_non_json_body():
    """An HTML error page from a proxy must not crash the mapper."""
    exc = addon.CloudHTTPError(502, '<html>Bad Gateway</html>')
    assert exc.body == {}
    assert exc.error_kind == ''
    code, _, retry = addon._translate_http_error(exc, BASE, 'upload')
    assert code == addon.ERR_NETWORK_UNAVAILABLE and retry is True


# ---------------------------------------------------------------------------
# Multipart encoder
# ---------------------------------------------------------------------------
def test_multipart_round_trips_binary_payload():
    """Binary bytes must survive verbatim — including CRLF and the `--`
    sequences that could be mistaken for a boundary."""
    payload = b'\x00\x01\r\n--not-a-boundary--\r\n\xff\xfe'
    body, ctype = addon._encode_multipart({}, {'audio': ('a.opus', payload, 'audio/ogg')})
    assert ctype.startswith('multipart/form-data; boundary=')
    boundary = ctype.split('boundary=')[1]
    assert payload in body
    assert body.endswith(f'--{boundary}--\r\n'.encode())
    assert b'filename="a.opus"' in body
    assert b'Content-Type: audio/ogg' in body


# ---------------------------------------------------------------------------
# End-to-end over stdio against a fake cloud
# ---------------------------------------------------------------------------
class _FakeCloud(ThreadingHTTPServer):
    """Records what the add-on sent and replays a scripted job lifecycle."""

    daemon_threads = True

    def __init__(self, *, polls_before_complete=2, completion=None, fail=None):
        super().__init__(('127.0.0.1', 0), _FakeCloudHandler)
        self.polls_before_complete = polls_before_complete
        self.completion = completion or {}
        self.fail = fail or {}          # stage -> (status, body dict)
        self.seen = {'auth': [], 'submit': None, 'upload_bytes': 0,
                     'polls': 0, 'cancelled': False, 'user_agents': []}

    @property
    def base_url(self):
        host, port = self.server_address[:2]
        return f'http://{host}:{port}'


class _FakeCloudHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):  # keep pytest output clean
        pass

    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _maybe_fail(self, stage) -> bool:
        spec = self.server.fail.get(stage)
        if not spec:
            return False
        status, payload = spec
        self._json(status, payload)
        return True

    def do_POST(self):
        srv = self.server
        srv.seen['auth'].append(self.headers.get('Authorization'))
        srv.seen['user_agents'].append(self.headers.get('User-Agent'))
        length = int(self.headers.get('Content-Length') or 0)
        raw = self.rfile.read(length) if length else b''

        if self.path == '/api/v1/asr/upload':
            if self._maybe_fail('upload'):
                return
            srv.seen['upload_bytes'] = len(raw)
            assert b'name="audio"' in raw, 'field name is part of the API contract'
            self._json(200, {'audio_url': 'https://upstream.test/audio.opus'})
            return

        if self.path == '/api/v1/asr/transcribe':
            if self._maybe_fail('submit'):
                return
            srv.seen['submit'] = json.loads(raw.decode())
            self._json(200, {'id': 'job-123'})
            return

        if self.path.endswith('/cancel'):
            srv.seen['cancelled'] = True
            self._json(200, {'status': 'cancelled'})
            return

        self._json(404, {'error': 'not_found'})

    def do_GET(self):
        srv = self.server
        if not self.path.startswith('/api/v1/jobs/'):
            self._json(404, {'error': 'not_found'})
            return
        if self._maybe_fail('poll'):
            return
        srv.seen['polls'] += 1
        if srv.seen['polls'] <= srv.polls_before_complete:
            self._json(200, {'status': 'processing', 'progress': 40})
            return
        self._json(200, dict(srv.completion, status='complete'))


class _Addon:
    """Drives the add-on as a real subprocess and collects its frames."""

    def __init__(self, env_extra):
        env = dict(os.environ, **env_extra)
        env['PYTHONPATH'] = str(REPO_ROOT)
        env['PYTHONUNBUFFERED'] = '1'
        self.proc = subprocess.Popen(
            [sys.executable, '-m', 'assemblyai_addon'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, cwd=str(REPO_ROOT),
        )

    def send(self, frame):
        self.proc.stdin.write((json.dumps(frame) + '\n').encode())
        self.proc.stdin.flush()

    def read_frame(self, timeout=30.0):
        # readline() on a pipe can't be given a timeout directly; a reader
        # thread keeps a hung add-on from wedging the whole suite.
        box = {}

        def _read():
            box['line'] = self.proc.stdout.readline()

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout)
        line = box.get('line')
        if not line:
            raise AssertionError(
                f'no frame within {timeout}s; stderr tail:\n{self._stderr_tail()}')
        return json.loads(line.decode())

    def read_until(self, ftype, timeout=60.0):
        """Collect frames until one of `ftype` arrives; return (it, all)."""
        deadline = time.monotonic() + timeout
        collected = []
        while time.monotonic() < deadline:
            frame = self.read_frame(timeout=max(1.0, deadline - time.monotonic()))
            collected.append(frame)
            if frame.get('type') == ftype:
                return frame, collected
        raise AssertionError(f'never saw {ftype}; got {collected}')

    def _stderr_tail(self):
        try:
            self.proc.stderr.flush()
        except Exception:
            pass
        return ''

    def close(self):
        try:
            self.send({'type': 'shutdown'})
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
        finally:
            for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
                try:
                    stream.close()
                except Exception:
                    pass


@pytest.fixture
def cloud():
    server = _FakeCloud()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def wav(tmp_path):
    """One second of silence, as a real file for ffmpeg to encode."""
    if not HAS_FFMPEG:
        pytest.skip('ffmpeg not installed')
    path = tmp_path / 'in.wav'
    subprocess.run(
        ['ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=16000:cl=mono',
         '-t', '1', str(path)],
        check=True, capture_output=True,
    )
    return str(path)


def test_handshake_and_shutdown():
    """hello must arrive before anything else and advertise the task."""
    a = _Addon({})
    try:
        hello = a.read_frame()
        assert hello['type'] == 'hello'
        assert hello['addon'] == 'assemblyai'
        assert hello['protocol'] == addon.PROTOCOL_VERSION
        tasks = [c['task'] for c in hello['capabilities']]
        assert tasks == ['asr.transcribe']
        a.send({'type': 'ready', 'host': 'test', 'host_version': '0'})
    finally:
        a.close()
    assert a.proc.returncode == 0


def test_missing_api_key_is_actionable():
    a = _Addon({'ASSEMBLYAI_API_KEY': '', 'SUBTITLD_CLOUD_API_KEY': ''})
    try:
        a.read_frame()  # hello
        a.send({'type': 'ready'})
        a.send({'type': 'asr.transcribe', 'id': 'r1',
                'params': {'audio_path': __file__, 'language': 'en'}})
        err, _ = a.read_until('error')
        assert err['code'] == addon.ERR_BAD_PARAMS
        assert 'API key' in err['message']
        # The message must tell the user where to get one.
        assert 'dashboard/keys' in err['message']
    finally:
        a.close()


def test_missing_audio_file_is_bad_params():
    a = _Addon({'ASSEMBLYAI_API_KEY': 'k'})
    try:
        a.read_frame()
        a.send({'type': 'ready'})
        a.send({'type': 'asr.transcribe', 'id': 'r1',
                'params': {'audio_path': '/definitely/not/here.wav'}})
        err, _ = a.read_until('error')
        assert err['code'] == addon.ERR_BAD_PARAMS
    finally:
        a.close()


def test_unknown_frame_still_answers():
    """An unanswered request id would leave the host's future pending until
    its own timeout."""
    a = _Addon({})
    try:
        a.read_frame()
        a.send({'type': 'ready'})
        a.send({'type': 'tts.synthesize', 'id': 'r9', 'params': {}})
        err, _ = a.read_until('error')
        assert err['id'] == 'r9'
        assert err['code'] == addon.ERR_BAD_PARAMS
    finally:
        a.close()


@needs_ffmpeg
def test_full_transcription_round_trip(cloud, wav):
    """The whole chain: encode → upload → submit → poll → partials → result."""
    cloud.completion = {
        'audio_duration': 10.0,
        'sentences': [
            {'start': 0, 'end': 2000, 'text': 'First line.', 'speaker': 'A'},
            {'start': 2100, 'end': 4000, 'text': 'Second line.', 'speaker': 'B'},
        ],
    }
    a = _Addon({'ASSEMBLYAI_API_KEY': 'test-key', 'ASSEMBLYAI_BASE_URL': cloud.base_url})
    try:
        a.read_frame()
        a.send({'type': 'ready'})
        a.send({'type': 'asr.transcribe', 'id': 'r1',
                'params': {'audio_path': wav, 'language': 'pt-br',
                           'options': {'model': 'nano'}}})
        result, frames = a.read_until('result')
    finally:
        a.close()

    segments = result['data']['segments']
    assert [s['text'] for s in segments] == ['First line.', 'Second line.']
    assert [s['speaker'] for s in segments] == ['A', 'B']
    assert segments[0]['end'] == pytest.approx(2.0)  # ms → s

    # One partial per cue, all before the result.
    partials = [f for f in frames if f.get('type') == 'partial']
    assert len(partials) == 2
    assert frames.index(partials[-1]) < frames.index(result)

    # Progress must be monotonic and end at 1.0 — a bar that goes backwards
    # is worse than no bar.
    values = [f['value'] for f in frames if f.get('type') == 'progress']
    assert values == sorted(values)
    assert values[-1] == pytest.approx(1.0)

    # The wire contract the cloud depends on.
    assert cloud.seen['auth'][0] == 'Bearer test-key'
    assert cloud.seen['submit']['model'] == 'subtitld-cloud:assemblyai/nano'
    assert cloud.seen['submit']['audio_url'] == 'https://upstream.test/audio.opus'
    assert cloud.seen['submit']['language'] == 'pt'  # locale trimmed to 2 letters
    assert cloud.seen['upload_bytes'] > 0
    # The stdlib's default UA gets 403'd by the WAF in front of the cloud.
    assert all('Python-urllib' not in (ua or '') for ua in cloud.seen['user_agents'])


@needs_ffmpeg
def test_slice_off_merges_speaker_turns_end_to_end(cloud, wav):
    cloud.completion = {
        'audio_duration': 10.0,
        'sentences': [
            {'start': 0, 'end': 2000, 'text': 'One.', 'speaker': 'A'},
            {'start': 2000, 'end': 4000, 'text': 'Two.', 'speaker': 'A'},
        ],
    }
    a = _Addon({'ASSEMBLYAI_API_KEY': 'k', 'ASSEMBLYAI_BASE_URL': cloud.base_url})
    try:
        a.read_frame()
        a.send({'type': 'ready'})
        a.send({'type': 'asr.transcribe', 'id': 'r1',
                'params': {'audio_path': wav, 'language': 'en',
                           'options': {'slice_by_phrases': False}}})
        result, _ = a.read_until('result')
    finally:
        a.close()
    assert [s['text'] for s in result['data']['segments']] == ['One. Two.']
    assert cloud.seen['submit']['slice_by_phrases'] is False


@needs_ffmpeg
def test_cancel_stops_polling_and_releases_the_hold(cloud, wav):
    """A cancel must reach the cloud — otherwise the credit hold leaks."""
    cloud.polls_before_complete = 10_000  # never completes on its own
    a = _Addon({'ASSEMBLYAI_API_KEY': 'k', 'ASSEMBLYAI_BASE_URL': cloud.base_url})
    try:
        a.read_frame()
        a.send({'type': 'ready'})
        a.send({'type': 'asr.transcribe', 'id': 'r1',
                'params': {'audio_path': wav, 'language': 'en'}})
        # Wait until it is genuinely polling before cancelling.
        deadline = time.monotonic() + 30
        while cloud.seen['polls'] < 1 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert cloud.seen['polls'] >= 1, 'add-on never reached the polling stage'

        a.send({'type': 'cancel', 'target': 'r1'})
        err, _ = a.read_until('error', timeout=30)
        assert err['code'] == addon.ERR_CANCELLED
    finally:
        a.close()
    assert cloud.seen['cancelled'] is True


@needs_ffmpeg
@pytest.mark.parametrize('stage,status,body,expect_code,needle', [
    ('upload', 401, {}, addon.ERR_BAD_PARAMS, 'Invalid API key'),
    ('submit', 402, {}, addon.ERR_BAD_PARAMS, 'top up'),
    ('submit', 404, {}, addon.ERR_MODEL_MISSING, 'Unknown model'),
    ('upload', 503, {'error': 'provider_not_configured'},
     addon.ERR_NETWORK_UNAVAILABLE, 'not configured'),
])
def test_cloud_errors_surface_to_the_host(cloud, wav, stage, status, body,
                                          expect_code, needle):
    cloud.fail = {stage: (status, body)}
    a = _Addon({'ASSEMBLYAI_API_KEY': 'k', 'ASSEMBLYAI_BASE_URL': cloud.base_url})
    try:
        a.read_frame()
        a.send({'type': 'ready'})
        a.send({'type': 'asr.transcribe', 'id': 'r1',
                'params': {'audio_path': wav, 'language': 'en'}})
        err, _ = a.read_until('error')
    finally:
        a.close()
    assert err['code'] == expect_code
    assert needle in err['message']


@needs_ffmpeg
def test_server_side_job_failure_is_not_a_crash(cloud, wav):
    """A failed job is an expected outcome — it must not read as
    'worker crashed', which would hide the server's actual reason."""
    cloud.polls_before_complete = 0
    cloud.completion = {}

    # Override the completion to report failure instead.
    class _FailingHandler(_FakeCloudHandler):
        def do_GET(self):
            self.server.seen['polls'] += 1
            self._json(200, {'status': 'error', 'error': 'upstream said no'})

    cloud.RequestHandlerClass = _FailingHandler

    a = _Addon({'ASSEMBLYAI_API_KEY': 'k', 'ASSEMBLYAI_BASE_URL': cloud.base_url})
    try:
        a.read_frame()
        a.send({'type': 'ready'})
        a.send({'type': 'asr.transcribe', 'id': 'r1',
                'params': {'audio_path': wav, 'language': 'en'}})
        err, _ = a.read_until('error')
    finally:
        a.close()
    assert err['code'] == addon.ERR_INTERNAL
    assert 'worker crashed' not in err['message']
    assert 'upstream said no' in err['message']
