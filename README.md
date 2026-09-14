# AssemblyAI ASR add-on for Subtitld

Cloud **speech-to-text** for [Subtitld](https://github.com/Subtitld/subtitld),
powered by [AssemblyAI](https://www.assemblyai.com/) and routed through
**Subtitld Cloud**.

- Serves the `asr.transcribe` task over Subtitld's add-on protocol.
- High-accuracy transcription with **speaker diarization**, in 20+ languages.
- Tiny download (~10 MB) and no models to fetch — the work happens server-side.

## How it works

The add-on never talks to AssemblyAI directly and never holds an AssemblyAI
key. It authenticates to **Subtitld Cloud** with *your* Subtitld Cloud key, and
the cloud relays to AssemblyAI using its own credentials:

```
Subtitld ──stdio──▶ this add-on ──HTTPS──▶ Subtitld Cloud ──▶ AssemblyAI
```

That indirection is deliberate: signature/webhook verification, credit
hold-and-settle, and upstream retry handling all live server-side, so this
process stays a small stdlib-only HTTP client.

One request runs: **encode** (ffmpeg → mono 24 kbps Opus) → **upload** →
**submit** → **poll every 2 s** → **emit cues**. Cancelling tells the cloud to
cancel too, so the credit hold is released rather than left hanging.

## Requirements

> [!IMPORTANT]
> **A Subtitld Cloud API key is required** — not an AssemblyAI key. Generate one
> at <https://cloud.subtitld.org/dashboard/keys/> and paste it into the add-on's
> **Configure** dialog. The key is shared with Subtitld's other cloud-backed
> providers, so you only enter it once.

> [!IMPORTANT]
> **This add-on sends your audio to a remote server.** The audio is re-encoded
> and uploaded to Subtitld Cloud, which forwards it to AssemblyAI for
> processing. Don't use it for material you can't send off-machine — Subtitld
> ships offline engines (Vosk, whisper.cpp) for that.

Transcription consumes Subtitld Cloud credit. An empty balance returns a clear
"Insufficient balance" error with a top-up link rather than failing silently.

`ffmpeg` is used to encode the audio; the add-on resolves it from
`SUBTITLD_FFMPEG_EXECUTABLE` (set by Subtitld) or from `PATH`.

## Usage

Install it from Subtitld's **Add-ons** dialog, configure your API key, then pick
**AssemblyAI (Subtitld Cloud)** as the transcription engine.

## Options

| Option | Default | Notes |
|--------|---------|-------|
| `api_key` | — | **Required.** Your Subtitld Cloud key. |
| `model` | `subtitld-cloud:assemblyai/best` | `best` (most accurate) or `nano` (faster, cheaper, more languages). A bare `best`/`nano` is accepted and expanded. |
| `speaker_labels` | `true` | Diarization — label each cue with who spoke it. |
| `slice_by_phrases` | `true` | On: one cue per sentence. Off: one cue per speaker turn. |
| `base_url` | `https://cloud.subtitld.org` | Advanced — point at a draft or self-hosted cloud. |

Each can also be set per request through the host's `options`, or via the
environment (`ASSEMBLYAI_API_KEY`, `ASSEMBLYAI_MODEL`, …). Resolution order is
**request option → `ASSEMBLYAI_*` → `SUBTITLD_CLOUD_*`**, so the shared cloud
credential slot works without re-entering the key here.

## Segmentation

The add-on **never re-segments by punctuation client-side** — it trusts the
server's own segmentation, preferring the richest shape the cloud returns:

| Shape returned | Result |
|----------------|--------|
| `sentences[]` | One cue per sentence (or per speaker turn with `slice_by_phrases` off). |
| `utterances[]` | One cue per speaker turn. `slice_by_phrases` has no effect. |
| `words[]` | Grouped by pause (0.6 s), speaker change, and runaway caps only. |
| `text` only | A single whole-audio cue, so there's still something to edit. |

Times arrive in milliseconds or seconds depending on the cloud build; the
add-on sniffs which by comparing against the known audio duration rather than
hard-coding a unit.

## Error codes

`bad_params` (no/invalid API key, insufficient balance, missing ffmpeg, bad
audio path), `model_missing` (unknown model — refresh the catalog),
`network_unavailable` (cloud unreachable, rate limited, or a 5xx — retryable),
`cancelled`, `internal` (job failed server-side, timeout, or anything else).

The human-readable `message` carries the specific cause, including the cloud's
typed 503 discriminators (`upload_failed`, `provider_not_configured`,
`upload_rejected_by_provider`) and any `upstream_status`.

## Development

```bash
pip install -e '.[dev]'
python -m assemblyai_addon    # runs the add-on (speaks the protocol on stdio)
pytest                        # protocol, translation, and end-to-end tests
```

The suite drives the add-on as a **real subprocess** against a local fake cloud
server, so the actual HTTP path, the hand-rolled multipart encoder, the polling
loop, and cancellation are all exercised. Tests that need `ffmpeg` skip
themselves when it isn't installed; CI installs it so they always run.

Build the bundle:

```bash
pip install -e '.[build]'
pyinstaller pyinstaller.spec --noconfirm   # -> dist/assemblyai-addon/
```

Releases are built for Linux/macOS/Windows by the `Release` GitHub Action on a
`v*` tag and published to the Subtitld add-ons catalog.

## Known limitations

- **Online only.** No offline fallback; a dropped connection fails the request.
- **Partials arrive at the end.** AssemblyAI's batch API produces cues only when
  the job completes, so `partial` frames come as one burst rather than
  progressively. The add-on advertises `streaming: false` rather than promise a
  cadence it can't keep — progress is still reported throughout.
- **15-minute ceiling** per request. A longer job keeps running server-side, but
  this process stops waiting for it.

## License

Apache-2.0 (this add-on). AssemblyAI's service is governed by its own terms.
