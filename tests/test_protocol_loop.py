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

import contextlib
import json
import os
import shutil
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
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


def test_404_with_html_body_blames_the_endpoint_not_the_model():
    """The incident case: pointing at a host that serves no API.

    Cloudflare/the origin answers 404 with an HTML page. Reporting that as
    "unknown model" sends the user to refresh a catalog that was never the
    problem, so a non-JSON 404 must blame base_url instead — at EVERY stage.
    """
    html = '<!doctype html><html><head><title>Not Found</title></head></html>'
    for stage in ('upload', 'submit', 'poll'):
        exc = addon.CloudHTTPError(404, html)
        code, msg, retry = addon._translate_http_error(exc, BASE, stage)
        assert code == addon.ERR_BAD_PARAMS, (stage, code)
        assert 'No Subtitld Cloud API at' in msg, (stage, msg)
        assert BASE in msg
        assert 'model' not in msg.lower(), (stage, msg)
        assert retry is False


def test_404_on_upload_with_json_body_is_not_a_model_error():
    """Upload does not involve a model, so it must never say one is unknown."""
    exc = addon.CloudHTTPError(404, json.dumps({'error': 'not_found'}))
    code, msg, _ = addon._translate_http_error(exc, BASE, 'upload')
    assert code != addon.ERR_MODEL_MISSING
    assert 'model' not in msg.lower(), msg


def test_429_names_the_stage_and_endpoint():
    """A bare "rate limited" hid a dead-endpoint problem for a whole session."""
    exc = addon.CloudHTTPError(429, '')
    code, msg, retry = addon._translate_http_error(exc, BASE, 'upload')
    assert code == addon.ERR_NETWORK_UNAVAILABLE and retry is True
    assert 'upload' in msg and BASE in msg


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
    scheme = 'http'

    def __init__(self, *, polls_before_complete=2, completion=None, fail=None,
                 poll_transient_status=0, poll_transient_count=0,
                 retry_after=None):
        super().__init__(('127.0.0.1', 0), _FakeCloudHandler)
        self.polls_before_complete = polls_before_complete
        self.completion = completion or {}
        self.fail = fail or {}          # stage -> (status, body dict)
        # Transient poll failures: the first `poll_transient_count` polls
        # answer `poll_transient_status` before normal service resumes.
        self.poll_transient_status = poll_transient_status
        self.poll_transient_count = poll_transient_count
        self.retry_after = retry_after
        self.seen = {'auth': [], 'submit': None, 'upload_bytes': 0,
                     'polls': 0, 'cancelled': False, 'user_agents': [],
                     'poll_failures_served': 0}

    @property
    def base_url(self):
        host, port = self.server_address[:2]
        return f'{self.scheme}://{host}:{port}'


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
        if srv.seen['poll_failures_served'] < srv.poll_transient_count:
            srv.seen['poll_failures_served'] += 1
            body = b'{"error":"slow down"}'
            self.send_response(srv.poll_transient_status)
            self.send_header('Content-Type', 'application/json')
            if srv.retry_after is not None:
                self.send_header('Retry-After', str(srv.retry_after))
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
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


@needs_ffmpeg
def test_transient_429_during_polling_does_not_lose_the_job(cloud, wav):
    """The incident: the cloud transcribed and BILLED the job, then a single
    429 mid-poll made the add-on discard the finished result.

    By the time we poll, the money is spent and the job id lives only in this
    process — so a transient failure must be ridden out, not treated as fatal.
    """
    cloud.poll_transient_status = 429
    cloud.poll_transient_count = 3
    cloud.retry_after = 1
    cloud.completion = {
        'audio_duration': 10.0,
        'sentences': [{'start': 0, 'end': 2000, 'text': 'Survived the 429.'}],
    }
    a = _Addon({'ASSEMBLYAI_API_KEY': 'k', 'ASSEMBLYAI_BASE_URL': cloud.base_url})
    try:
        a.read_frame()
        a.send({'type': 'ready'})
        a.send({'type': 'asr.transcribe', 'id': 'r1',
                'params': {'audio_path': wav, 'language': 'en'}})
        result, _ = a.read_until('result', timeout=120)
    finally:
        a.close()
    assert [s['text'] for s in result['data']['segments']] == ['Survived the 429.']
    assert cloud.seen['poll_failures_served'] == 3, 'the 429s never fired'


@needs_ffmpeg
def test_transient_5xx_during_polling_is_also_survived(cloud, wav):
    cloud.poll_transient_status = 503
    cloud.poll_transient_count = 2
    cloud.completion = {'audio_duration': 5.0, 'text': 'Still here.'}
    a = _Addon({'ASSEMBLYAI_API_KEY': 'k', 'ASSEMBLYAI_BASE_URL': cloud.base_url})
    try:
        a.read_frame()
        a.send({'type': 'ready'})
        a.send({'type': 'asr.transcribe', 'id': 'r1',
                'params': {'audio_path': wav, 'language': 'en'}})
        result, _ = a.read_until('result', timeout=120)
    finally:
        a.close()
    assert result['data']['segments'][0]['text'] == 'Still here.'


@needs_ffmpeg
def test_sustained_poll_failure_warns_the_job_may_have_completed(cloud, wav):
    """Giving up is allowed — silently implying the work is gone is not.

    The user has been charged; the message must send them to the dashboard
    rather than straight to paying for a re-run.
    """
    cloud.poll_transient_status = 429
    cloud.poll_transient_count = 10_000      # never recovers
    a = _Addon({'ASSEMBLYAI_API_KEY': 'k', 'ASSEMBLYAI_BASE_URL': cloud.base_url})
    try:
        a.read_frame()
        a.send({'type': 'ready'})
        a.send({'type': 'asr.transcribe', 'id': 'r1',
                'params': {'audio_path': wav, 'language': 'en'}})
        err, _ = a.read_until('error', timeout=180)
    finally:
        a.close()
    assert 'may have finished on the server' in err['message'], err['message']
    assert cloud.seen['poll_failures_served'] >= addon._POLL_MAX_CONSECUTIVE_FAILURES - 1


def test_retry_after_is_read_from_the_header():
    exc = addon.CloudHTTPError(429, '{}', {'Retry-After': '7'})
    assert exc.retry_after == 7.0
    assert exc.transient is True
    # A date-form or junk Retry-After must not explode; we fall back to backoff.
    assert addon.CloudHTTPError(429, '{}', {'Retry-After': 'Wed, 21 Oct 2026 07:28:00 GMT'}).retry_after == 0.0
    assert addon.CloudHTTPError(429, '{}', {}).retry_after == 0.0


def test_only_429_and_5xx_count_as_transient():
    """A 401 or 404 must still fail fast — retrying those just wastes time."""
    for status in (400, 401, 402, 404, 413):
        assert addon.CloudHTTPError(status, '{}').transient is False, status
    for status in (429, 500, 502, 503, 504):
        assert addon.CloudHTTPError(status, '{}').transient is True, status


def test_cloud_endpoint_is_not_a_user_setting():
    """Which server to talk to is a developer concern, set for a whole session
    via SUBTITLD_CLOUD_BASE_URL — not a field in the Configure panel. It was one
    in 1.0.0/1.0.1 and was removed on request."""
    manifest = json.loads((REPO_ROOT / 'manifest.json').read_text(encoding='utf-8'))
    keys = {f['key'] for f in manifest['config_schema']['fields']}
    assert 'base_url' not in keys, keys


def test_errors_never_point_at_the_removed_setting():
    """Messages must tell users about the env var that actually exists."""
    html = '<!doctype html><title>Not Found</title>'
    for exc, stage in ((addon.CloudHTTPError(404, html), 'upload'),
                       (addon.CloudHTTPError(429, ''), 'poll')):
        _, msg, _ = addon._translate_http_error(exc, BASE, stage)
        assert 'Cloud endpoint setting' not in msg, msg
        assert 'SUBTITLD_CLOUD_BASE_URL' in msg, msg


# ---------------------------------------------------------------------------
# ffmpeg must not read the protocol channel
# ---------------------------------------------------------------------------
# Stand-in for ffmpeg/ffprobe. Once the test has queued host frames it takes
# every byte its stdin offers — the worst case of ffmpeg's keyboard polling,
# which eats queued frames whenever it shares the add-on's stdin.
_STDIN_EATER = '''
import json, os, select, sys, time
from pathlib import Path

tool = Path(sys.argv[0]).name
work = Path(os.environ['STUB_FFMPEG_DIR'])
(work / 'started').touch()
deadline = time.monotonic() + 10
while not (work / 'go').exists() and time.monotonic() < deadline:
    time.sleep(0.01)
consumed = 0
while select.select([0], [], [], 0.5)[0]:
    chunk = os.read(0, 4096)
    if not chunk:
        break
    consumed += len(chunk)
(work / 'report.json').write_text(json.dumps({'argv': sys.argv[1:], 'consumed': consumed}))
if tool == 'ffprobe':
    if '-nostdin' in sys.argv:  # the real ffprobe rejects it too
        sys.exit("Failed to set value for option 'nostdin': Option not found")
    print('1.5')
else:
    Path(sys.argv[-1]).write_bytes(b'OggS stub')
'''


@contextlib.contextmanager
def _host_pipe_as_stdin():
    """Put a fresh pipe on fd 0, where host frames arrive; yield its write end."""
    read_fd, write_fd = os.pipe()
    saved = os.dup(0)
    os.dup2(read_fd, 0)
    os.close(read_fd)
    host = os.fdopen(write_fd, 'wb')
    try:
        yield host
    finally:
        host.close()
        os.dup2(saved, 0)
        os.close(saved)


def _wait_until_started(marker, job):
    deadline = time.monotonic() + 10
    while not marker.exists():
        if job.done():
            job.result()  # surfaces the helper's own error
            raise AssertionError('helper returned without running the stand-in')
        assert time.monotonic() < deadline, 'stand-in never started'
        time.sleep(0.01)


@pytest.mark.skipif(sys.platform == 'win32', reason='the stand-in is a shebang script')
@pytest.mark.parametrize('tool', ['ffmpeg', 'ffprobe'])
def test_ffmpeg_does_not_swallow_queued_host_frames(tool, tmp_path, monkeypatch):
    """A `cancel` sent while audio encodes must still reach the dispatcher.

    In-process on purpose: through the `_Addon` harness the dispatcher thread
    races ffmpeg for each frame, so a regression would only fail sometimes.
    Here the stand-in is the only other reader, so it fails every time.
    """
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    for name in ('ffmpeg', 'ffprobe'):  # ffprobe is looked up beside ffmpeg
        stub = bindir / name
        stub.write_text(f'#!{sys.executable}\n{_STDIN_EATER}', encoding='utf-8')
        stub.chmod(0o755)
    monkeypatch.setenv('SUBTITLD_FFMPEG_EXECUTABLE', str(bindir / 'ffmpeg'))
    monkeypatch.setenv('STUB_FFMPEG_DIR', str(tmp_path))
    src = tmp_path / 'in.wav'
    src.write_bytes(b'RIFF')
    if tool == 'ffmpeg':
        def run():
            return addon._encode_opus(str(src), str(tmp_path / 'out.opus'))
    else:
        def run():
            return addon._probe_duration(str(src))

    sent = [
        {'type': 'cancel', 'target': 'r1'},
        {'type': 'asr.transcribe', 'id': 'r2', 'params': {'audio_path': str(src)}},
        {'type': 'cancel', 'target': 'r2'},
        {'type': 'shutdown'},
    ]
    received = []
    with _host_pipe_as_stdin() as host, ThreadPoolExecutor(1) as pool:
        job = pool.submit(run)
        _wait_until_started(tmp_path / 'started', job)
        for frame in sent:
            host.write((json.dumps(frame) + '\n').encode())
        host.close()
        (tmp_path / 'go').touch()
        result = job.result(timeout=30)

        with open(0, encoding='utf-8', closefd=False) as stdin:
            monkeypatch.setattr(sys, 'stdin', stdin)
            while (frame := addon._read_frame_blocking()) is not None:
                received.append(frame)

    assert received == sent, 'host frames were lost while the tool ran'
    report = json.loads((tmp_path / 'report.json').read_text(encoding='utf-8'))
    assert report['consumed'] == 0, 'the tool was handed the protocol stdin'
    if tool == 'ffmpeg':
        assert '-nostdin' in report['argv']
    else:
        # ffprobe has no -nostdin, and _probe_duration reads any failure as 0.0.
        assert result == 1.5


# ---------------------------------------------------------------------------
# TLS: the frozen build's OpenSSL may find no system CA store at all
# ---------------------------------------------------------------------------
HAS_OPENSSL = shutil.which('openssl') is not None


@pytest.fixture
def fresh_https_context():
    """`_https_context` is cached per process; these tests change its inputs."""
    addon._https_context.cache_clear()
    yield
    addon._https_context.cache_clear()


def _no_system_ca_store(monkeypatch, tmp_path):
    """What the frozen OpenSSL sees on a machine laid out unlike the build host."""
    monkeypatch.setenv('SSL_CERT_FILE', str(tmp_path / 'no-such-ca-file.pem'))
    monkeypatch.setenv('SSL_CERT_DIR', str(tmp_path / 'no-such-ca-dir'))


@pytest.fixture(scope='module')
def test_ca(tmp_path_factory):
    """A throwaway CA, and a certificate for 127.0.0.1 signed by it.

    certifi has never heard of this CA, so trusting it proves the system
    store is still consulted alongside certifi.
    """
    if not HAS_OPENSSL:
        pytest.skip('openssl CLI not installed')
    d = tmp_path_factory.mktemp('tls')

    def openssl(*args):
        subprocess.run(['openssl', *args], cwd=d, check=True, capture_output=True)

    openssl('req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '2',
            '-subj', '/CN=Subtitld add-on test CA',
            '-addext', 'basicConstraints=critical,CA:TRUE',
            '-addext', 'keyUsage=critical,keyCertSign,cRLSign',
            '-keyout', 'ca.key', '-out', 'ca.pem')
    openssl('req', '-newkey', 'rsa:2048', '-nodes', '-subj', '/CN=127.0.0.1',
            '-keyout', 'server.key', '-out', 'server.csr')
    # Python 3.13+ verifies in X509-strict mode, which wants these extensions.
    (d / 'server.ext').write_text(
        'basicConstraints=critical,CA:FALSE\n'
        'keyUsage=critical,digitalSignature,keyEncipherment\n'
        'extendedKeyUsage=serverAuth\n'
        'subjectAltName=IP:127.0.0.1\n'
        'subjectKeyIdentifier=hash\n'
        'authorityKeyIdentifier=keyid,issuer\n')
    openssl('x509', '-req', '-in', 'server.csr', '-CA', 'ca.pem', '-CAkey', 'ca.key',
            '-set_serial', '1', '-days', '2', '-extfile', 'server.ext',
            '-out', 'server.pem')
    return d


@pytest.fixture
def tls_cloud(test_ca):
    """The fake cloud behind HTTPS, with a certificate from `test_ca`."""
    server = _FakeCloud()
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(test_ca / 'server.pem', test_ca / 'server.key')
    # Handshakes run inside accept(), where a client's rejection is dropped
    # quietly instead of printing a traceback from a handler thread.
    server.socket = ctx.wrap_socket(server.socket, server_side=True)
    server.scheme = 'https'
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.mark.skipif(sys.platform == 'win32',
                    reason='Python on Windows reads the system store directly')
def test_certifi_is_trusted_when_the_system_store_is_missing(
        fresh_https_context, monkeypatch, tmp_path):
    import certifi

    _no_system_ca_store(monkeypatch, tmp_path)
    assert ssl.create_default_context().cert_store_stats()['x509_ca'] == 0, \
        'precondition: the env vars must hide the system store'
    bundled = ssl.create_default_context(cafile=certifi.where())
    assert (addon._https_context().cert_store_stats()['x509_ca']
            == bundled.cert_store_stats()['x509_ca'])


def test_certificate_errors_are_not_reported_as_connection_problems(
        fresh_https_context, tls_cloud, monkeypatch, tmp_path):
    _no_system_ca_store(monkeypatch, tmp_path)
    with pytest.raises(urllib.error.URLError) as info:
        addon._request(f'{tls_cloud.base_url}/api/v1/jobs/x', api_key='k')
    code, message, retryable = addon._translate_url_error(
        info.value, tls_cloud.base_url, 'Could not reach Subtitld Cloud')
    assert code == addon.ERR_NETWORK_UNAVAILABLE
    assert 'security certificate of 127.0.0.1' in message
    assert 'date and time' in message
    assert 'Could not reach' not in message
    assert retryable is False


def test_plain_network_errors_keep_the_connection_message():
    exc = urllib.error.URLError(ConnectionRefusedError(111, 'Connection refused'))
    code, message, retryable = addon._translate_url_error(
        exc, BASE, 'Could not reach Subtitld Cloud')
    assert code == addon.ERR_NETWORK_UNAVAILABLE
    assert message.startswith('Could not reach Subtitld Cloud:')
    assert retryable is True


def test_poll_gives_up_at_once_on_a_certificate_error(monkeypatch):
    """Retrying a certificate that won't verify only delays the real message."""
    calls = []

    def _refuse(url, **kwargs):
        calls.append(url)
        raise urllib.error.URLError(
            ssl.SSLCertVerificationError(1, 'certificate verify failed'))

    monkeypatch.setattr(addon, '_request', _refuse)
    monkeypatch.setattr(addon, '_emit', lambda frame: None)
    with pytest.raises(urllib.error.URLError):
        addon._poll_until_done(addon._WorkerState('r1'), 'job-1',
                               base_url=BASE, api_key='k')
    assert len(calls) == 1


@needs_ffmpeg
def test_system_trusted_ca_still_verifies_end_to_end(tls_cloud, test_ca, wav, tmp_path):
    """certifi is added to the system store, not swapped in for it.

    `cafile=certifi.where()` alone would reject a CA only the system trusts,
    such as a corporate proxy's. Here SSL_CERT_FILE plays the system store
    and holds a CA certifi does not know.
    """
    tls_cloud.completion = {'audio_duration': 1.0, 'text': 'Over TLS.'}
    a = _Addon({'ASSEMBLYAI_API_KEY': 'k', 'ASSEMBLYAI_BASE_URL': tls_cloud.base_url,
                'SSL_CERT_FILE': str(test_ca / 'ca.pem'),
                'SSL_CERT_DIR': str(tmp_path / 'no-such-ca-dir')})
    try:
        a.read_frame()
        a.send({'type': 'ready'})
        a.send({'type': 'asr.transcribe', 'id': 'r1',
                'params': {'audio_path': wav, 'language': 'en'}})
        result, _ = a.read_until('result')
    finally:
        a.close()
    assert [s['text'] for s in result['data']['segments']] == ['Over TLS.']


@needs_ffmpeg
def test_untrusted_certificate_reaches_the_host_as_such(tls_cloud, wav, tmp_path):
    a = _Addon({'ASSEMBLYAI_API_KEY': 'k', 'ASSEMBLYAI_BASE_URL': tls_cloud.base_url,
                'SSL_CERT_FILE': str(tmp_path / 'no-such-ca-file.pem'),
                'SSL_CERT_DIR': str(tmp_path / 'no-such-ca-dir')})
    try:
        a.read_frame()
        a.send({'type': 'ready'})
        a.send({'type': 'asr.transcribe', 'id': 'r1',
                'params': {'audio_path': wav, 'language': 'en'}})
        err, _ = a.read_until('error')
    finally:
        a.close()
    assert err['code'] == addon.ERR_NETWORK_UNAVAILABLE
    assert 'security certificate' in err['message']
    assert 'Could not reach' not in err['message']
    assert err['retryable'] is False
    assert tls_cloud.seen['upload_bytes'] == 0


# ---------------------------------------------------------------------------
# ffmpeg must not inherit the frozen bundle's library path
# ---------------------------------------------------------------------------
# Stand-in for ffmpeg/ffprobe that refuses to run with the bundle on
# LD_LIBRARY_PATH — the real ones would load the bundle's older libssl and
# libz, and fail to start on a distro newer than the build runner.
_LIBRARY_PATH_CHECKER = '''
import json, os, sys
from pathlib import Path

tool = Path(sys.argv[0]).name
seen = os.environ.get('LD_LIBRARY_PATH')
work = Path(os.environ['STUB_FFMPEG_DIR'])
(work / f'{tool}.json').write_text(json.dumps({'LD_LIBRARY_PATH': seen}))
if seen is not None and os.environ['STUB_BUNDLE_LIBS'] in seen.split(os.pathsep):
    sys.exit(f'started with the bundle on LD_LIBRARY_PATH: {seen}')
if tool == 'ffprobe':
    print('1.5')
else:
    Path(sys.argv[-1]).write_bytes(b'OggS stub')
'''


@pytest.mark.skipif(not sys.platform.startswith('linux'),
                    reason='PyInstaller rewrites LD_LIBRARY_PATH on Linux only')
@pytest.mark.parametrize('users_value', [None, '/opt/users/lib'])
def test_ffmpeg_does_not_inherit_the_bundle_library_path(
        users_value, tmp_path, monkeypatch):
    """Frozen, the bootloader has put `_internal` first on LD_LIBRARY_PATH
    and saved any earlier value in LD_LIBRARY_PATH_ORIG. ffmpeg must see the
    user's value again, or no variable at all if they had none."""
    bundle_libs = tmp_path / 'assemblyai-addon' / '_internal'
    bundle_libs.mkdir(parents=True)
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    for name in ('ffmpeg', 'ffprobe'):
        stub = bindir / name
        stub.write_text(f'#!{sys.executable}\n{_LIBRARY_PATH_CHECKER}', encoding='utf-8')
        stub.chmod(0o755)

    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    if users_value is None:
        monkeypatch.setenv('LD_LIBRARY_PATH', str(bundle_libs))
        monkeypatch.delenv('LD_LIBRARY_PATH_ORIG', raising=False)
    else:
        monkeypatch.setenv('LD_LIBRARY_PATH', f'{bundle_libs}{os.pathsep}{users_value}')
        monkeypatch.setenv('LD_LIBRARY_PATH_ORIG', users_value)
    monkeypatch.setenv('SUBTITLD_FFMPEG_EXECUTABLE', str(bindir / 'ffmpeg'))
    monkeypatch.setenv('STUB_FFMPEG_DIR', str(tmp_path))
    monkeypatch.setenv('STUB_BUNDLE_LIBS', str(bundle_libs))
    src = tmp_path / 'in.wav'
    src.write_bytes(b'RIFF')

    # _probe_duration reads any ffprobe failure as 0.0.
    assert addon._probe_duration(str(src)) == 1.5
    addon._encode_opus(str(src), str(tmp_path / 'out.opus'))  # raises if ffmpeg failed

    for tool in ('ffmpeg', 'ffprobe'):
        report = json.loads((tmp_path / f'{tool}.json').read_text(encoding='utf-8'))
        assert report['LD_LIBRARY_PATH'] == users_value, tool


def test_unfrozen_runs_pass_the_environment_through(monkeypatch):
    """From source there is no bundle, and a developer's LD_LIBRARY_PATH is
    theirs to keep."""
    monkeypatch.delattr(sys, 'frozen', raising=False)
    assert addon._tool_env() is None
