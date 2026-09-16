"""AssemblyAI (via Subtitld Cloud) ASR add-on — JSON-line protocol entry-point.

Wire format
-----------
Stdin and stdout carry one JSON object per line. Stderr is reserved for
free-form logging — the host prefixes each line with
`[addon:org.subtitld.assemblyai]` and surfaces it verbatim.

What this add-on is
-------------------
A thin, **stdlib-only** client for Subtitld Cloud's ASR relay. It never
talks to AssemblyAI directly and never holds an upstream provider key:
the cloud stores that, and the desktop authenticates to the *cloud* with
the user's own Subtitld Cloud key. That indirection is the entire point —
signature/webhook verification, credit hold-and-settle, and upstream 5xx
retries all live server-side, so this process stays small enough to freeze
into a ~10 MB bundle with no third-party wheels.

Request lifecycle (one `asr.transcribe`)
----------------------------------------
1. **Encode.** ffmpeg → mono 24 kbps Opus-in-Ogg. ~10 MB/hour, which keeps
   the upload inside any sane cap without chunked-transfer machinery.
2. **Upload.** `POST /api/v1/asr/upload` (multipart, field name `audio`)
   → `{"audio_url": "..."}`. The cloud forwards the bytes upstream with
   its own key and hands back a fetchable URL.
3. **Submit.** `POST /api/v1/asr/transcribe` with
   `{model, audio_url, language, speaker_labels, slice_by_phrases}`
   → `{"id": "<job>"}`.
4. **Poll.** `GET /api/v1/jobs/<id>` every 2 s until `status == "complete"`
   (or `error`/`failed`). A `cancel` frame fires
   `POST /api/v1/jobs/<id>/cancel` so the cloud releases the credit hold
   instead of leaking it.
5. **Parse + emit.** Translate the completion payload into cues, emit one
   `partial` per cue, then a terminal `result`.

Why polling and not a webhook: the desktop has no reachable address. The
cloud *does* use webhooks upstream — that's why `status` flips without us
having to re-ask AssemblyAI ourselves — but the last hop to a user's
laptop is necessarily a pull.

Segmentation
------------
We never re-segment client-side by punctuation. The cloud relays whatever
AssemblyAI produced and we trust it, preferring the richest shape present:
`sentences[]` → `utterances[]` → `words[]` → bare `text`. See
`_segments_from_job` for the full decision table. This mirrors the legacy
in-app plugin, which delegated entirely to AssemblyAI's `get_sentences()`
and matched user expectation exactly.

Configuration
-------------
Resolved per request, first hit wins:

  1. `params['options']` — per-request overrides from the host.
  2. `ASSEMBLYAI_*` env — the add-on's own **Configure** dialog. The host
     maps config keys to `<ADDON_ID>_<KEY>` env vars (see
     `addon_provider._build_addon_env`).
  3. `SUBTITLD_CLOUD_*` env — the **shared** cloud credential slot, so a
     user who already pasted their key for another cloud-backed provider
     doesn't have to paste it again.

Threading
---------
One worker thread per request, same shape as the Vosk add-on. The work is
almost entirely blocking I/O (ffmpeg, socket reads, `time.sleep` between
polls), so a thread is both simpler and cheaper than dragging asyncio into
a frozen bundle. The host serialises ASR requests to one provider at a
time regardless.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

ADDON_ID = 'assemblyai'
ADDON_VERSION = '1.0.2'
PROTOCOL_VERSION = 1

# The cloud namespaces every provider it relays. Part of the API contract —
# model ids look like `subtitld-cloud:assemblyai/best`.
PUBLIC_ID_PREFIX = 'subtitld-cloud:'
UPSTREAM_PROVIDER_ID = 'assemblyai'

PRODUCTION_BASE_URL = 'https://cloud.subtitld.org'
DRAFT_BASE_URL = 'https://draft-cloud.subtitld.org'

# Cloudflare's bot protection in front of the cloud rejects the stdlib's
# default `Python-urllib/<v>` UA with a 403 before the request ever reaches
# the origin — a perfectly valid key then reads as "no models available".
# Sending a real UA both clears the WAF and gives the cloud team a usage
# signal per add-on version.
USER_AGENT = f'Subtitld-addon-assemblyai/{ADDON_VERSION}'

# ---------------------------------------------------------------------------
# Error codes. Mirrors `subtitld.modules.addons.protocol.ErrorCode` — inlined
# so the frozen binary takes no runtime dependency on the host package.
# ---------------------------------------------------------------------------
ERR_BAD_PARAMS = 'bad_params'
ERR_INTERNAL = 'internal'
ERR_CANCELLED = 'cancelled'
ERR_NETWORK_UNAVAILABLE = 'network_unavailable'
ERR_MODEL_MISSING = 'model_missing'

log = logging.getLogger('assemblyai_addon')

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
# Polling starts responsive and backs off. A flat 2 s was both wasteful and
# self-defeating: a 30-minute file meant hundreds of requests against the same
# endpoint, which is what tripped the cloud's rate limiter in the first place.
_POLL_INTERVAL_START_SEC = 2.0
_POLL_INTERVAL_MAX_SEC = 20.0
_POLL_INTERVAL_GROWTH = 1.35

# The job is already paid for by the time we are polling, so give up late
# rather than early — an abandoned poll loop bins work the user was billed for.
_POLL_TIMEOUT_SEC = 60 * 60  # 1 hour

# How many CONSECUTIVE transient failures (429 / 5xx / network) end the poll.
# One blip must never kill a finished job; a genuinely dead endpoint still
# terminates in well under a minute of retries.
_POLL_MAX_CONSECUTIVE_FAILURES = 6
_HTTP_TIMEOUT_SEC = 120.0
_UPLOAD_TIMEOUT_SEC = 600.0  # a long file at a bad uplink is still just one POST

# Word-grouping thresholds for the deep fallback path (fires only when the
# cloud surfaces NEITHER sentences nor utterances — nothing but a flat
# `words[]`). No punctuation detection: the desktop deliberately does not try
# to out-guess the server's own segmentation. These are runaway guards, not
# a segmentation strategy.
_WORD_GAP_BREAK_SEC = 0.6
# ~500 chars is about three dense lines; 30 s exceeds any natural single
# sentence by a wide margin. Both sit far above normal speech so they never
# fire mid-thought.
_PHRASE_HARD_MAX_CHARS = 500
_PHRASE_HARD_MAX_DURATION_SEC = 30.0

# Languages AssemblyAI supports for the `best`/`nano` tiers. Advertised in
# `hello` so the host can filter the engine list by the project language.
_SUPPORTED_LANGUAGES = [
    'en', 'en-us', 'en-gb', 'en-au',
    'es', 'fr', 'de', 'it', 'pt', 'pt-br', 'nl', 'hi', 'ja', 'zh',
    'fi', 'ko', 'pl', 'ru', 'tr', 'uk', 'vi',
]


# ---------------------------------------------------------------------------
# Frame encode / emit. Inlined (no host import) so the binary stays standalone.
# ---------------------------------------------------------------------------
_stdout_lock = threading.Lock()


def _emit(frame: dict) -> None:
    """Write one frame to stdout. Thread-safe — worker threads share it."""
    line = json.dumps(frame, ensure_ascii=False, separators=(',', ':')) + '\n'
    data = line.encode('utf-8')
    with _stdout_lock:
        try:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
        except (BrokenPipeError, OSError):
            # Host went away; the main loop exits on the next empty readline().
            pass


def _emit_progress(req_id: str, value: float, message: str = '') -> None:
    _emit({'id': req_id, 'type': 'progress',
           'value': max(0.0, min(1.0, value)), 'message': message})


def _emit_partial(req_id: str, segment: dict) -> None:
    _emit({'id': req_id, 'type': 'partial', 'data': {'segment': segment}})


def _emit_result(req_id: str, data: dict) -> None:
    _emit({'id': req_id, 'type': 'result', 'data': data})


def _emit_error(req_id: str, code: str, message: str, retryable: bool = False) -> None:
    _emit({'id': req_id, 'type': 'error', 'code': code,
           'message': message, 'retryable': retryable})


# ---------------------------------------------------------------------------
# Configuration resolution
# ---------------------------------------------------------------------------
def _env(*names: str) -> str:
    """First non-empty value among `names`, stripped. '' when none are set."""
    for name in names:
        value = (os.environ.get(name) or '').strip()
        if value:
            return value
    return ''


def _resolve_base_url(options: dict) -> str:
    """Active cloud endpoint, without a trailing slash.

    `SUBTITLD_CLOUD_BASE_URL` is the long-standing development override used
    by the rest of the app (it points a whole running instance at
    draft-cloud); honouring it here keeps the add-on consistent with the
    built-in cloud providers rather than needing its own separate switch.
    """
    raw = (
        str(options.get('base_url') or '').strip()
        or _env('ASSEMBLYAI_BASE_URL', 'SUBTITLD_CLOUD_BASE_URL')
        or PRODUCTION_BASE_URL
    )
    return raw.rstrip('/')


def _resolve_api_key(options: dict) -> str:
    """User's Subtitld Cloud key. '' when unconfigured — callers must error
    out cleanly rather than send an empty Bearer and get an opaque 401."""
    return (
        str(options.get('api_key') or '').strip()
        or _env('ASSEMBLYAI_API_KEY', 'SUBTITLD_CLOUD_API_KEY')
    )


def _resolve_model(options: dict) -> str:
    """Public model id, e.g. `subtitld-cloud:assemblyai/best`.

    A bare tier name (`best`, `nano`) is accepted and expanded — the
    Configure dialog's dropdown stores full public ids, but a hand-edited
    config or a per-request override is easy to write short, and silently
    404ing on that would be a poor trade.
    """
    model = (
        str(options.get('model') or '').strip()
        or _env('ASSEMBLYAI_MODEL')
    )
    if not model:
        return f'{PUBLIC_ID_PREFIX}{UPSTREAM_PROVIDER_ID}/best'
    if not model.startswith(PUBLIC_ID_PREFIX):
        if '/' not in model:
            model = f'{UPSTREAM_PROVIDER_ID}/{model}'
        model = f'{PUBLIC_ID_PREFIX}{model}'
    return model


def _opt_bool(options: dict, key: str, env_name: str, default: bool) -> bool:
    """Three-way resolve for a boolean: request option, env, then default.

    `options.get(key) or default` would be wrong — an explicit `False` is
    falsy and would silently become the default.
    """
    if key in options and options[key] is not None:
        return bool(options[key])
    raw = _env(env_name)
    if raw:
        return raw.strip().lower() not in ('0', 'false', 'no', 'off', '')
    return default


# ---------------------------------------------------------------------------
# HTTP. Stdlib only — `requests` would mean a wheel in the frozen bundle for
# what amounts to four calls.
# ---------------------------------------------------------------------------
class CloudHTTPError(Exception):
    """An HTTP error from the cloud, with the parsed body kept intact.

    The cloud puts machine-readable detail in the JSON body (`error`,
    `upstream_status`), and that detail is what turns a bare "503" into an
    actionable message. `urllib` only lets you read the body once, so we do
    it here, immediately, and carry the result.
    """

    def __init__(self, status: int, body_text: str, headers=None):
        self.status = int(status)
        self.body_text = body_text or ''
        self.headers = dict(headers or {})
        # `json_ok` tracks whether the body PARSED, which is not the same as
        # whether it has content: the API legitimately answers `{}`, and that
        # empty-but-valid dict is falsy. Conflating the two made a valid JSON
        # 404 look like a missing endpoint. An absent body stays `True` —
        # only a body that is present and unparseable (an HTML error page from
        # the edge, say) means we are not talking to the API at all.
        self.json_ok = True
        self.body = {}
        if self.body_text:
            try:
                parsed = json.loads(self.body_text)
            except (ValueError, TypeError):
                self.json_ok = False
            else:
                if isinstance(parsed, dict):
                    self.body = parsed
        super().__init__(f'HTTP {self.status}: {self.body_text[:300]}')

    @property
    def error_kind(self) -> str:
        """The cloud's typed error discriminator, e.g. `upload_failed`."""
        for key in ('error', 'code', 'detail'):
            value = self.body.get(key)
            if isinstance(value, str) and value:
                return value
        return ''

    @property
    def upstream_status(self) -> Any:
        return self.body.get('upstream_status')

    @property
    def retry_after(self) -> float:
        """Seconds the server asked us to wait, 0.0 when it didn't say.

        Only the delta-seconds form is honoured; the HTTP-date form is rare
        here and guessing at clock skew is worse than falling back to our own
        backoff.
        """
        for key, value in self.headers.items():
            if key.lower() != 'retry-after':
                continue
            try:
                return max(0.0, float(str(value).strip()))
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    @property
    def transient(self) -> bool:
        """True when retrying the SAME request could plausibly succeed.

        429 and 5xx are the cloud saying "not now", not "never". Treating them
        as fatal is what threw away completed, already-billed jobs.
        """
        return self.status == 429 or 500 <= self.status < 600


def _request(url: str, *, api_key: str, method: str = 'GET',
             data: bytes | None = None, content_type: str = '',
             timeout: float = _HTTP_TIMEOUT_SEC) -> dict:
    """One authenticated cloud call returning parsed JSON.

    Raises `CloudHTTPError` for any non-2xx (body preserved) and
    `urllib.error.URLError` for transport failures, which the caller maps to
    `network_unavailable`.
    """
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header('Authorization', f'Bearer {api_key}')
    req.add_header('Accept', 'application/json')
    req.add_header('User-Agent', USER_AGENT)
    if content_type:
        req.add_header('Content-Type', content_type)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode('utf-8', errors='replace')
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode('utf-8', errors='replace')
        except Exception:
            detail = ''
        raise CloudHTTPError(exc.code, detail, getattr(exc, 'headers', None)) from exc

    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise RuntimeError(f'cloud returned non-JSON body: {raw[:300]!r}') from exc
    return parsed if isinstance(parsed, dict) else {'data': parsed}


def _post_json(url: str, payload: dict, *, api_key: str,
               timeout: float = _HTTP_TIMEOUT_SEC) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    return _request(url, api_key=api_key, method='POST', data=body,
                    content_type='application/json', timeout=timeout)


def _encode_multipart(fields: dict[str, str],
                      files: dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
    """Build a `multipart/form-data` body. Returns `(body, content_type)`.

    Hand-rolled because the stdlib has no encoder and pulling `requests` into
    a frozen bundle for one POST is a bad trade. The boundary is a uuid4 hex,
    which cannot collide with binary payload content in any practical sense.
    """
    boundary = f'----SubtitldAddonBoundary{uuid.uuid4().hex}'
    out = bytearray()
    sep = f'--{boundary}\r\n'.encode('utf-8')

    for name, value in fields.items():
        out += sep
        out += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode('utf-8')
        out += str(value).encode('utf-8') + b'\r\n'

    for name, (filename, payload, ctype) in files.items():
        out += sep
        out += (
            f'Content-Disposition: form-data; name="{name}"; '
            f'filename="{filename}"\r\n'
        ).encode('utf-8')
        out += f'Content-Type: {ctype}\r\n\r\n'.encode('utf-8')
        out += payload + b'\r\n'

    out += f'--{boundary}--\r\n'.encode('utf-8')
    return bytes(out), f'multipart/form-data; boundary={boundary}'


# ---------------------------------------------------------------------------
# Error translation. The cloud's status codes carry specific, actionable
# meanings; a bare "HTTP 402" in the UI helps nobody.
#
# The host's ErrorCode enum has no auth/payment members, so everything the
# *user* must fix (bad key, empty balance, unknown model) maps to
# `bad_params` (non-retryable — retrying changes nothing), and everything
# transient maps to `network_unavailable` (retryable). The human-readable
# `message` carries the real distinction, and that is what the UI shows.
# ---------------------------------------------------------------------------
_UPLOAD_503_HINTS = {
    'upload_failed':
        'Subtitld Cloud could not forward the audio to AssemblyAI. '
        'This is usually transient — try again in a moment.',
    'provider_not_configured':
        'AssemblyAI is not configured on the Subtitld Cloud server. '
        'Contact support — nothing to fix on this machine.',
    'upload_rejected_by_provider':
        'AssemblyAI rejected the uploaded audio. The file may be corrupt, '
        'silent, or longer than the provider allows.',
}


def _translate_http_error(exc: CloudHTTPError, base_url: str,
                          stage: str) -> tuple[str, str, bool]:
    """Map a cloud HTTP error to `(code, message, retryable)`."""
    status = exc.status
    kind = exc.error_kind

    if status == 401:
        return (ERR_BAD_PARAMS,
                'Invalid API key. Check your Subtitld Cloud key in the '
                f'add-on settings, or generate a new one at {base_url}/dashboard/keys/',
                False)

    if status == 402:
        return (ERR_BAD_PARAMS,
                'Insufficient balance — top up at '
                f'{base_url}/dashboard/topup/',
                False)

    if status == 404:
        # A 404 means one of three unrelated things, and saying the wrong one
        # sends the user hunting the wrong problem.
        #
        # First: is the ENDPOINT itself missing? The API answers with JSON, so
        # a 404 carrying a non-JSON body (an HTML error page from the edge or
        # a bare origin) means we're talking to a host that doesn't serve this
        # API at all — wrong `base_url`, or a server that isn't deployed yet.
        # That is not a model problem and must not be reported as one.
        if not exc.json_ok:
            return (ERR_BAD_PARAMS,
                    f'No Subtitld Cloud API at {base_url} — the server returned '
                    f'"not found" for /api/v1/asr/. Set SUBTITLD_CLOUD_BASE_URL to a '
                    f'running Subtitld Cloud server and restart Subtitld.',
                    False)
        # Otherwise the API answered, and the 404 is about the thing we named.
        if stage == 'poll':
            return (ERR_INTERNAL,
                    'Subtitld Cloud no longer knows this job id. It may have '
                    'expired; run the transcription again.',
                    False)
        if stage == 'upload':
            return (ERR_INTERNAL,
                    'Subtitld Cloud rejected the upload as not found. This is '
                    'a server-side routing problem, not something to fix here.',
                    False)
        return (ERR_MODEL_MISSING,
                'Unknown model — refresh the catalog in the add-on settings '
                'and pick a model again.',
                False)

    if status == 413:
        return (ERR_BAD_PARAMS,
                'The encoded audio is too large for Subtitld Cloud to accept. '
                'Transcribe a shorter section.',
                False)

    if status == 429:
        return (ERR_NETWORK_UNAVAILABLE,
                f'Rate limited by Subtitld Cloud during {stage} (at {base_url}). '
                'Wait a moment and try again; if it repeats, check that '
                'SUBTITLD_CLOUD_BASE_URL points at a live server.',
                True)

    if status == 503:
        hint = _UPLOAD_503_HINTS.get(kind)
        if hint is None:
            hint = ('Subtitld Cloud is temporarily unavailable. '
                    'Try again in a moment.')
        upstream = exc.upstream_status
        if upstream is not None:
            hint = f'{hint} (upstream status {upstream})'
        # `provider_not_configured` is a server-side misconfiguration —
        # retrying on a timer would just hammer a broken endpoint.
        retryable = kind != 'provider_not_configured'
        return (ERR_NETWORK_UNAVAILABLE, hint, retryable)

    if 500 <= status < 600:
        return (ERR_NETWORK_UNAVAILABLE,
                f'Subtitld Cloud error during {stage} (HTTP {status}). '
                'Try again in a moment.',
                True)

    detail = exc.body_text[:300].strip()
    suffix = f' {detail}' if detail else ''
    return (ERR_INTERNAL, f'Subtitld Cloud {stage} failed (HTTP {status}).{suffix}', False)


# ---------------------------------------------------------------------------
# Response → segments translation
#
# The completion payload relays whatever AssemblyAI produced. We prefer the
# richest server-side segmentation available and never re-segment by
# punctuation client-side:
#
#   - Best: `sentences[]` (AssemblyAI's own sentence endpoint) → one cue per
#     sentence. `slice_by_phrases` picks between that and one cue per speaker
#     TURN (merge consecutive same-speaker sentences).
#   - Next: `utterances[]` (speaker turns) → one cue per turn. The slice
#     toggle is a no-op — without sentence data we don't invent boundaries.
#   - Fallback: `words[]` only → group by pause / runaway caps.
#   - Last resort: bare `text` → a single whole-audio cue, so the user has
#     something to edit rather than an empty import.
#
# Times are normalised to seconds. AssemblyAI's native unit is milliseconds
# but the cloud may already have converted, so we sniff rather than assume.
# ---------------------------------------------------------------------------
def _detect_ms_divisor(sample_value: float, audio_duration_sec: float) -> float:
    """Return 1000.0 if `sample_value` is likely milliseconds, else 1.0.

    If we know the audio duration and a sample time exceeds it by more than
    10x, the value must be ms — no honest reading has a single cue ending
    ten times past the end of the audio. Without a known duration, fall back
    to "larger than 24 h of seconds must be ms", which is correct for any
    recording anyone will actually transcribe.
    """
    if audio_duration_sec > 0 and sample_value > audio_duration_sec * 10:
        return 1000.0
    if sample_value > 86_400:
        return 1000.0
    return 1.0


def _segments_from_sentences(sentences: list, audio_duration_sec: float) -> list[dict]:
    """Translate `sentences[]` verbatim — one cue per sentence.

    The preferred path, and deliberately dumb: no regrouping, no punctuation
    logic. Trusting the server's segmentation is the whole point.
    """
    sample = max((s.get('end', 0) or 0 for s in sentences), default=0)
    divisor = _detect_ms_divisor(float(sample), audio_duration_sec)
    out: list[dict] = []
    for s in sentences:
        text = (s.get('text') or '').strip()
        if not text:
            continue
        out.append({
            'start': float(s.get('start', 0) or 0) / divisor,
            'end': float(s.get('end', 0) or 0) / divisor,
            'text': text,
            # Present per sentence when diarization is on; absent otherwise.
            'speaker': str(s.get('speaker', '') or ''),
        })
    return out


def _merge_consecutive_speakers(segments: list[dict]) -> list[dict]:
    """Collapse runs of same-speaker cues into one cue per speaker TURN.

    Used when slice-by-phrase is OFF. Empty-string speakers all compare
    equal, so an undiarized transcript collapses to one cue per contiguous
    run of text — which is exactly the coarse grain that toggle asks for.
    """
    if not segments:
        return []
    out: list[dict] = []
    current = dict(segments[0])
    for seg in segments[1:]:
        if seg.get('speaker', '') == current.get('speaker', ''):
            current['end'] = seg.get('end', current['end'])
            current['text'] = f"{current['text']} {seg.get('text', '')}".strip()
        else:
            out.append(current)
            current = dict(seg)
    out.append(current)
    return out


def _segments_from_utterances(utterances: list, audio_duration_sec: float) -> list[dict]:
    """Translate `utterances[]` — one cue per speaker turn, verbatim.

    No re-slicing. Sentence-grained cues require the cloud to surface
    `sentences`; without it we emit turns as-is rather than guessing.
    """
    sample = max((u.get('end', 0) or 0 for u in utterances), default=0)
    divisor = _detect_ms_divisor(float(sample), audio_duration_sec)
    out: list[dict] = []
    for u in utterances:
        text = (u.get('text') or '').strip()
        if not text:
            continue
        # AssemblyAI labels speakers A/B/C…, which already matches the rest
        # of the app's convention.
        out.append({
            'start': float(u.get('start', 0) or 0) / divisor,
            'end': float(u.get('end', 0) or 0) / divisor,
            'text': text,
            'speaker': str(u.get('speaker', '') or ''),
        })
    return out


def _segments_from_words(words: list, audio_duration_sec: float) -> list[dict]:
    """Deep fallback — fires only when neither sentences nor utterances exist.

    Groups on speaker change, long pause, and runaway caps. No punctuation
    detection: this is the minimum viable grouping that avoids one cue per
    word, not an attempt at segmentation.
    """
    if not words:
        return []
    sample = max((w.get('end', 0) or 0 for w in words), default=0)
    divisor = _detect_ms_divisor(float(sample), audio_duration_sec)

    out: list[dict] = []
    current: dict | None = None

    for w in words:
        text = (w.get('text') or '').strip()
        if not text:
            continue
        start = float(w.get('start', 0) or 0) / divisor
        end = float(w.get('end', 0) or 0) / divisor
        speaker = str(w.get('speaker', '') or '')

        if current is None:
            current = {'start': start, 'end': end, 'text': text, 'speaker': speaker}
            continue

        if speaker and current['speaker'] and speaker != current['speaker']:
            out.append(current)
            current = {'start': start, 'end': end, 'text': text, 'speaker': speaker}
            continue

        if start - current['end'] >= _WORD_GAP_BREAK_SEC:
            out.append(current)
            current = {'start': start, 'end': end, 'text': text, 'speaker': speaker}
            continue

        prospective_len = len(current['text']) + 1 + len(text)
        prospective_dur = end - current['start']
        if (prospective_len > _PHRASE_HARD_MAX_CHARS
                or prospective_dur > _PHRASE_HARD_MAX_DURATION_SEC):
            out.append(current)
            current = {'start': start, 'end': end, 'text': text, 'speaker': speaker}
            continue

        current['text'] += ' ' + text
        current['end'] = end
        if not current['speaker'] and speaker:
            current['speaker'] = speaker

    if current is not None:
        out.append(current)
    return out


def _segments_from_job(data: dict, *, fallback_duration_sec: float = 0.0,
                       slice_by_phrase: bool = True) -> list[dict]:
    """Pick the richest available shape and degrade gracefully.

    `fallback_duration_sec` is the locally-probed audio duration, used when
    the cloud omits `audio_duration` so the whole-audio fallback cue still
    lands with a real end time. Without it that cue would draw at zero width
    and divide-by-zero anything that reads cue durations.

    Returns `[]` only for a genuinely empty response, so the import view can
    surface "nothing recognised" cleanly instead of showing a blank cue.
    """
    audio_duration_sec = float(data.get('audio_duration') or 0.0)
    if audio_duration_sec > 86_400:  # the duration field itself may be ms
        audio_duration_sec = audio_duration_sec / 1000.0
    if audio_duration_sec <= 0 and fallback_duration_sec > 0:
        audio_duration_sec = fallback_duration_sec

    sentences = data.get('sentences') or []
    if sentences:
        out = _segments_from_sentences(sentences, audio_duration_sec)
        if out:
            return out if slice_by_phrase else _merge_consecutive_speakers(out)

    utterances = data.get('utterances') or []
    if utterances:
        out = _segments_from_utterances(utterances, audio_duration_sec)
        if out:
            return out

    words = data.get('words') or []
    if words:
        out = _segments_from_words(words, audio_duration_sec)
        if out:
            return out

    text = (data.get('text') or '').strip()
    if not text:
        return []
    return [{'start': 0.0, 'end': audio_duration_sec, 'text': text, 'speaker': ''}]


# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------
def _find_ffmpeg() -> str | None:
    return (
        os.environ.get('SUBTITLD_FFMPEG_EXECUTABLE')
        or os.environ.get('FFMPEG_EXECUTABLE')
        or shutil.which('ffmpeg')
    )


def _no_window_startupinfo():
    """Keep ffmpeg from flashing a console window on Windows GUI hosts."""
    if sys.platform != 'win32':
        return None
    si = subprocess.STARTUPINFO()  # type: ignore[attr-defined]
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore[attr-defined]
    return si


def _probe_duration(path: str) -> float:
    """Best-effort source duration in seconds, 0.0 when unknown.

    Only used as the fallback end time for the whole-audio cue, so a failure
    here degrades that one cue's length rather than the request.
    """
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return 0.0
    # ffprobe usually sits beside ffmpeg; if it doesn't, skip the probe
    # rather than shelling out to something unexpected.
    ffprobe = str(Path(ffmpeg).with_name(
        'ffprobe.exe' if sys.platform == 'win32' else 'ffprobe'))
    if not os.path.isfile(ffprobe):
        ffprobe = shutil.which('ffprobe') or ''
    if not ffprobe:
        return 0.0
    try:
        proc = subprocess.run(
            [ffprobe, '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', path],
            capture_output=True, text=True, timeout=30,
            startupinfo=_no_window_startupinfo(),
        )
        return float((proc.stdout or '').strip())
    except Exception:
        return 0.0


def _encode_opus(src: str, dst: str) -> None:
    """Re-encode to mono 24 kbps Opus-in-Ogg.

    24 kbps VBR at `-application voip` is transparent for speech and lands
    at roughly 10 MB per hour, so even a long interview uploads as a single
    modest POST. 48 kHz is Opus's native rate — anything else just makes the
    encoder resample internally.
    """
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(
            'ffmpeg not found. Set SUBTITLD_FFMPEG_EXECUTABLE or put ffmpeg on PATH.')

    proc = subprocess.run(
        [ffmpeg, '-y', '-i', src,
         '-c:a', 'libopus', '-b:a', '24k', '-vbr', 'on',
         '-compression_level', '10', '-application', 'voip',
         '-ac', '1', '-ar', '48000',
         dst],
        capture_output=True, text=True,
        startupinfo=_no_window_startupinfo(),
    )
    if proc.returncode != 0 or not os.path.isfile(dst) or os.path.getsize(dst) == 0:
        tail = (proc.stderr or '').strip().splitlines()[-3:]
        raise RuntimeError('audio re-encoding failed: ' + ' / '.join(tail))


# ---------------------------------------------------------------------------
# Worker plumbing
# ---------------------------------------------------------------------------
class _WorkerState:
    """Per-request handle shared between the dispatcher and its worker.

    `job_id` is written by the worker once the cloud accepts the submit, and
    read by the cancel path — hence the lock. Without it a cancel arriving in
    the window between submit and assignment would silently skip the
    server-side cancel and leak the credit hold.
    """

    def __init__(self, req_id: str):
        self.req_id = req_id
        self.cancel_flag = threading.Event()
        self.thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._job_id = ''
        self._base_url = ''
        self._api_key = ''

    def set_job(self, job_id: str, base_url: str, api_key: str) -> None:
        with self._lock:
            self._job_id = job_id
            self._base_url = base_url
            self._api_key = api_key

    def job(self) -> tuple[str, str, str]:
        with self._lock:
            return self._job_id, self._base_url, self._api_key


_active_requests_lock = threading.Lock()
_active_requests: dict[str, _WorkerState] = {}


def _spawn_worker(req_id: str, target, *args) -> None:
    state = _WorkerState(req_id)
    with _active_requests_lock:
        _active_requests[req_id] = state

    def _run():
        try:
            target(state, *args)
        except Exception:
            log.exception('worker for request %s crashed', req_id)
            tb = traceback.format_exc().strip().splitlines()[-1]
            _emit_error(req_id, ERR_INTERNAL, f'worker crashed: {tb}')
        finally:
            with _active_requests_lock:
                _active_requests.pop(req_id, None)

    t = threading.Thread(target=_run, name=f'assemblyai-worker-{req_id}', daemon=True)
    state.thread = t
    t.start()


def _cancel_cloud_job(state: _WorkerState) -> None:
    """Best-effort `POST /api/v1/jobs/<id>/cancel`.

    Failure is logged, never surfaced: the user already asked to stop, and a
    cancel that couldn't reach the server doesn't change what they see. The
    call matters because it releases the credit hold server-side.
    """
    job_id, base_url, api_key = state.job()
    if not job_id or not base_url or not api_key:
        return
    try:
        _request(f'{base_url}/api/v1/jobs/{urllib.parse.quote(job_id)}/cancel',
                 api_key=api_key, method='POST', timeout=15.0)
        log.info('cancelled cloud job %s', job_id)
    except Exception as exc:
        log.warning('could not cancel cloud job %s: %s', job_id, exc)


# ---------------------------------------------------------------------------
# The request itself
# ---------------------------------------------------------------------------
def _handle_transcribe(state: _WorkerState, params: dict) -> None:
    req_id = state.req_id
    audio_path = params.get('audio_path')
    language = (params.get('language') or '').strip().lower()
    options = params.get('options') or {}
    if not isinstance(options, dict):
        options = {}

    if not audio_path or not os.path.isfile(audio_path):
        _emit_error(req_id, ERR_BAD_PARAMS,
                    f'audio_path missing or not a file: {audio_path!r}')
        return

    base_url = _resolve_base_url(options)
    api_key = _resolve_api_key(options)
    model = _resolve_model(options)
    speaker_labels = _opt_bool(options, 'speaker_labels', 'ASSEMBLYAI_SPEAKER_LABELS', True)
    slice_by_phrases = _opt_bool(options, 'slice_by_phrases', 'ASSEMBLYAI_SLICE_BY_PHRASES', True)

    if not api_key:
        _emit_error(req_id, ERR_BAD_PARAMS,
                    'No Subtitld Cloud API key configured. Add one in the add-on '
                    f'settings — generate a key at {base_url}/dashboard/keys/')
        return

    if _find_ffmpeg() is None:
        _emit_error(req_id, ERR_BAD_PARAMS,
                    'ffmpeg not found. Set SUBTITLD_FFMPEG_EXECUTABLE or put '
                    'ffmpeg on PATH.')
        return

    log.info('transcribe: model=%s language=%r diarize=%s slice=%s base=%s',
             model, language, speaker_labels, slice_by_phrases, base_url)

    # `mkdtemp` rather than NamedTemporaryFile: ffmpeg needs to *create* the
    # output itself, and on Windows it cannot open a path another process
    # still holds. The whole directory is removed in `finally`.
    workdir = tempfile.mkdtemp(prefix='subtitld-assemblyai-')
    opus_path = os.path.join(workdir, 'audio.opus')
    try:
        # -- 1. encode ------------------------------------------------------
        _emit_progress(req_id, 0.05, 'Encoding audio')
        fallback_duration = _probe_duration(audio_path)
        try:
            _encode_opus(audio_path, opus_path)
        except Exception as exc:
            _emit_error(req_id, ERR_INTERNAL, str(exc))
            return

        if state.cancel_flag.is_set():
            _emit_error(req_id, ERR_CANCELLED, 'cancelled')
            return

        # -- 2. upload ------------------------------------------------------
        size_mb = os.path.getsize(opus_path) / (1024 * 1024)
        _emit_progress(req_id, 0.15, f'Uploading audio ({size_mb:.1f} MB)')
        try:
            audio_url = _upload(opus_path, base_url=base_url, api_key=api_key)
        except CloudHTTPError as exc:
            code, message, retryable = _translate_http_error(exc, base_url, 'upload')
            _emit_error(req_id, code, message, retryable)
            return
        except urllib.error.URLError as exc:
            _emit_error(req_id, ERR_NETWORK_UNAVAILABLE,
                        f'Could not reach Subtitld Cloud: {exc.reason}', True)
            return
        except Exception as exc:
            _emit_error(req_id, ERR_INTERNAL, f'Upload failed: {exc}')
            return

        if state.cancel_flag.is_set():
            _emit_error(req_id, ERR_CANCELLED, 'cancelled')
            return

        # -- 3. submit ------------------------------------------------------
        _emit_progress(req_id, 0.3, 'Submitting transcription job')
        payload = {
            'model': model,
            'audio_url': audio_url,
            # AssemblyAI wants a bare two-letter tag; the host may hand us
            # a locale like `pt-br`.
            'language': language[:2] if language else '',
            'speaker_labels': bool(speaker_labels),
            'slice_by_phrases': bool(slice_by_phrases),
        }
        try:
            response = _post_json(f'{base_url}/api/v1/asr/transcribe',
                                  payload, api_key=api_key)
        except CloudHTTPError as exc:
            code, message, retryable = _translate_http_error(exc, base_url, 'submit')
            _emit_error(req_id, code, message, retryable)
            return
        except urllib.error.URLError as exc:
            _emit_error(req_id, ERR_NETWORK_UNAVAILABLE,
                        f'Could not reach Subtitld Cloud: {exc.reason}', True)
            return

        job_id = str(response.get('id') or '')
        if not job_id:
            _emit_error(req_id, ERR_INTERNAL,
                        f'Cloud response missing job id: {response!r}')
            return
        state.set_job(job_id, base_url, api_key)
        log.info('submitted job %s', job_id)

        # -- 4. poll --------------------------------------------------------
        try:
            job = _poll_until_done(state, job_id, base_url=base_url, api_key=api_key)
        except CloudHTTPError as exc:
            code, message, retryable = _translate_http_error(exc, base_url, 'poll')
            _emit_error(req_id, code, message, retryable)
            return
        except urllib.error.URLError as exc:
            _emit_error(req_id, ERR_NETWORK_UNAVAILABLE,
                        f'Lost contact with Subtitld Cloud: {exc.reason}', True)
            return
        except RuntimeError as exc:
            # Job failed server-side, or the 15-minute ceiling hit. Both are
            # expected outcomes with a useful message already attached — they
            # must not fall through to the generic "worker crashed" handler.
            _emit_error(req_id, ERR_INTERNAL, str(exc))
            return
        if job is None:
            # Cancelled — the poll loop already told the cloud to stop.
            _emit_error(req_id, ERR_CANCELLED, 'cancelled')
            return

        # -- 5. parse + emit -------------------------------------------------
        _emit_progress(req_id, 0.95, 'Building subtitles')
        segments = _segments_from_job(
            job,
            fallback_duration_sec=fallback_duration,
            slice_by_phrase=slice_by_phrases,
        )
        for seg in segments:
            _emit_partial(req_id, seg)
        _emit_progress(req_id, 1.0, '')
        _emit_result(req_id, {'segments': segments, 'language': language})
        log.info('job %s complete: %d segment(s)', job_id, len(segments))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _upload(opus_path: str, *, base_url: str, api_key: str) -> str:
    """POST the encoded audio to the cloud's upload proxy, return `audio_url`.

    The cloud forwards to AssemblyAI's own upload endpoint using its stored
    upstream key, so the desktop never holds a provider credential — the
    whole reason this path is cloud-routed.

    The file is read fully into memory: at 24 kbps that's ~10 MB/hour, well
    inside anything reasonable, and skipping chunked-streaming machinery
    keeps the failure modes down to "timeout" and "server said no".
    """
    with open(opus_path, 'rb') as fh:
        audio_bytes = fh.read()

    # ffmpeg writes `.opus` as Ogg-encapsulated Opus (RFC 7845); `audio/ogg`
    # is the registered type and the upstream accepts it directly, so the
    # cloud proxy never has to remux.
    ctype = mimetypes.guess_type(opus_path)[0] or 'audio/ogg'
    body, content_type = _encode_multipart(
        fields={},
        # Field name `audio` is the cloud API contract — do not rename
        # without a coordinated cloud API bump.
        files={'audio': (os.path.basename(opus_path), audio_bytes, ctype)},
    )
    response = _request(f'{base_url}/api/v1/asr/upload', api_key=api_key,
                        method='POST', data=body, content_type=content_type,
                        timeout=_UPLOAD_TIMEOUT_SEC)

    audio_url = str(response.get('audio_url') or '')
    if not audio_url:
        # Surface a shape change loudly rather than submitting an empty URL
        # and getting an inscrutable upstream failure later.
        raise RuntimeError(f'upload response missing audio_url: {response!r}')
    return audio_url


def _poll_until_done(state: _WorkerState, job_id: str, *,
                     base_url: str, api_key: str) -> dict | None:
    """Poll `/api/v1/jobs/<id>` until terminal. Returns None if cancelled.

    Transient failures do NOT end the job. By the time we are polling, the
    audio is uploaded and the transcription is running and billed — bailing
    out on one 429 or one 502 throws away work the user has already paid for
    and cannot get back, because the job id dies with this function. So 429
    and 5xx and network blips are retried with backoff, honouring
    `Retry-After` when the server sends one, and only
    `_POLL_MAX_CONSECUTIVE_FAILURES` failures IN A ROW give up.

    The interval itself grows from `_POLL_INTERVAL_START_SEC` toward
    `_POLL_INTERVAL_MAX_SEC`. A flat short interval is what provoked the rate
    limiting: a long file meant hundreds of identical requests. Backing off
    keeps the first few seconds responsive for short clips while making a
    long job cost a couple of dozen requests instead of hundreds.

    Progress maps to 0.3-0.9 using the job's own `progress` when the cloud
    reports one; otherwise it creeps asymptotically toward 0.9, because a
    frozen bar reads as a hang and we have no honest estimate to show.
    """
    req_id = state.req_id
    url = f'{base_url}/api/v1/jobs/{urllib.parse.quote(job_id)}'
    deadline = time.monotonic() + _POLL_TIMEOUT_SEC
    interval = _POLL_INTERVAL_START_SEC
    polls = 0
    consecutive_failures = 0
    last_error = ''

    def _sleep(seconds: float) -> bool:
        """Sleep in slices; False if cancelled partway through."""
        slept = 0.0
        while slept < seconds:
            if state.cancel_flag.is_set():
                return False
            time.sleep(min(0.1, seconds - slept))
            slept += 0.1
        return True

    while True:
        if state.cancel_flag.is_set():
            _cancel_cloud_job(state)
            return None

        if time.monotonic() > deadline:
            _cancel_cloud_job(state)
            raise RuntimeError(
                f'Transcription still unfinished after '
                f'{int(_POLL_TIMEOUT_SEC // 60)} minutes; giving up waiting. '
                'The job may still complete on the server.')

        try:
            job = _request(url, api_key=api_key)
        except CloudHTTPError as exc:
            if not exc.transient:
                raise
            consecutive_failures += 1
            last_error = f'HTTP {exc.status}'
            if consecutive_failures >= _POLL_MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f'Lost contact with Subtitld Cloud while waiting for the '
                    f'transcription ({last_error}, {consecutive_failures} tries). '
                    'The job may have finished on the server — check your '
                    'dashboard before paying to run it again.') from exc
            wait = exc.retry_after or min(interval * 2, _POLL_INTERVAL_MAX_SEC)
            log.warning('poll %s failed (%s); retry %d/%d in %.1fs',
                        job_id, last_error, consecutive_failures,
                        _POLL_MAX_CONSECUTIVE_FAILURES, wait)
            _emit_progress(req_id, 0.3 + 0.6 * min(1.0, polls / 60.0),
                           f'Waiting for Subtitld Cloud ({last_error})')
            if not _sleep(wait):
                _cancel_cloud_job(state)
                return None
            continue
        except urllib.error.URLError as exc:
            # Same reasoning as above: a dropped connection is not a reason to
            # discard a running, paid-for job.
            consecutive_failures += 1
            last_error = str(getattr(exc, 'reason', exc))
            if consecutive_failures >= _POLL_MAX_CONSECUTIVE_FAILURES:
                raise
            wait = min(interval * 2, _POLL_INTERVAL_MAX_SEC)
            log.warning('poll %s network error (%s); retry %d/%d in %.1fs',
                        job_id, last_error, consecutive_failures,
                        _POLL_MAX_CONSECUTIVE_FAILURES, wait)
            if not _sleep(wait):
                _cancel_cloud_job(state)
                return None
            continue

        consecutive_failures = 0
        status = str(job.get('status') or '').lower()
        polls += 1

        if status in ('complete', 'completed', 'done', 'success'):
            return job

        if status in ('error', 'failed'):
            message = (job.get('error') or job.get('message')
                       or 'Transcription failed on the server.')
            raise RuntimeError(f'Subtitld Cloud: {message}')

        reported = job.get('progress')
        if isinstance(reported, (int, float)) and reported > 0:
            # Accept either 0-1 or 0-100; the cloud has used both.
            frac = float(reported) / (100.0 if reported > 1 else 1.0)
            value = 0.3 + 0.6 * max(0.0, min(1.0, frac))
        else:
            value = 0.9 - 0.6 * (0.94 ** polls)
        _emit_progress(req_id, value, f'Transcribing ({status or "queued"})')

        if not _sleep(interval):
            _cancel_cloud_job(state)
            return None
        interval = min(interval * _POLL_INTERVAL_GROWTH, _POLL_INTERVAL_MAX_SEC)


# ---------------------------------------------------------------------------
# Handshake + main loop
# ---------------------------------------------------------------------------
def _send_hello() -> None:
    _emit({
        'type': 'hello',
        'protocol': PROTOCOL_VERSION,
        'addon': ADDON_ID,
        'version': ADDON_VERSION,
        'capabilities': [
            {
                'task': 'asr.transcribe',
                'languages': _SUPPORTED_LANGUAGES,
                # Cues only exist once the job completes, so `partial` frames
                # arrive as a burst at the end rather than progressively.
                # Advertising streaming would promise a cadence we can't keep.
                'streaming': False,
                'voice_clone': False,
            },
        ],
    })


def _send_hello_error(message: str, code: str = ERR_INTERNAL) -> None:
    _emit({'type': 'hello_error', 'code': code, 'message': message})


def _handle_cancel(target_id: str) -> None:
    with _active_requests_lock:
        state = _active_requests.get(target_id)
    if state is None:
        log.info('cancel: no active request %s', target_id)
        return
    log.info('cancel: signalling worker for %s', target_id)
    state.cancel_flag.set()


def _read_frame_blocking() -> dict | None:
    line = sys.stdin.buffer.readline()
    if not line:
        return None  # stdin closed → host went away
    try:
        text = line.decode('utf-8', errors='replace').strip().lstrip('\ufeff')
        if not text:
            return {}
        return json.loads(text)
    except json.JSONDecodeError as exc:
        log.warning('malformed frame %r: %s', line, exc)
        return {}


def _handshake() -> bool:
    """hello/ready handshake. True when the host accepted us."""
    if _find_ffmpeg() is None:
        # Not fatal here — we still want to appear in the engine list and
        # give a clear error on the first request, rather than vanishing
        # from the UI with no explanation.
        log.warning('ffmpeg not found at startup — requests will fail until one is on PATH')

    _send_hello()

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        frame = _read_frame_blocking()
        if frame is None:
            return False
        if not frame:
            continue
        if frame.get('type') == 'ready':
            log.info('handshake complete; host=%s host_version=%s',
                     frame.get('host'), frame.get('host_version'))
            return True
        log.warning('unexpected pre-ready frame: %r', frame)

    log.warning('host did not send ready within 30s')
    return False


def _main_loop() -> int:
    while True:
        frame = _read_frame_blocking()
        if frame is None:
            log.info('stdin closed; exiting')
            return 0
        if not frame:
            continue
        ftype = frame.get('type')

        if ftype == 'shutdown':
            log.info('shutdown requested')
            with _active_requests_lock:
                states = list(_active_requests.values())
            for state in states:
                state.cancel_flag.set()
            for state in states:
                if state.thread is not None:
                    state.thread.join(timeout=4.0)
            return 0

        if ftype == 'cancel':
            target = frame.get('target')
            if isinstance(target, str):
                _handle_cancel(target)
            continue

        if ftype == 'asr.transcribe':
            req_id = frame.get('id')
            params = frame.get('params') or {}
            if not isinstance(req_id, str) or not req_id:
                log.warning('asr.transcribe without id: %r', frame)
                continue
            if not isinstance(params, dict):
                _emit_error(req_id, ERR_BAD_PARAMS, 'params must be an object')
                continue
            _spawn_worker(req_id, _handle_transcribe, params)
            continue

        # Unknown task — still answer so the host's future resolves instead
        # of hanging until its timeout.
        req_id = frame.get('id')
        if isinstance(req_id, str) and req_id:
            _emit_error(req_id, ERR_BAD_PARAMS, f'unsupported frame type {ftype!r}')


def main() -> int:
    logging.basicConfig(
        level=os.environ.get('ASSEMBLYAI_ADDON_LOG_LEVEL', 'INFO').upper(),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        stream=sys.stderr,
    )
    log.info('assemblyai-addon %s starting on %s %s (python %s)',
             ADDON_VERSION, platform.system(), platform.machine(),
             platform.python_version())

    if not _handshake():
        return 1

    try:
        return _main_loop()
    except KeyboardInterrupt:
        log.info('interrupted')
        return 130
    except Exception:
        log.exception('main loop crashed')
        return 1


if __name__ == '__main__':
    sys.exit(main())
