# arabic-diarized-transcribe

Turns one Arabic recording with two or more speakers into a speaker-labelled transcript with timestamps.
It handles dialects such as Gulf/Najdi, plus English words mixed in.
One script, no UI, no database.

## How it works

Cohere Transcribe returns no timestamps and no speaker labels, so the order is fixed:

1. `ffmpeg` converts the input (m4a, mp3, wav, ...) to 16 kHz mono wav.
2. `pyannote/speaker-diarization-community-1` diarizes it. The script uses `exclusive_speaker_diarization` (one speaker at any moment). `--max-speakers` defaults to 3.
3. Turn cleanup: consecutive turns from the same speaker with a gap under 0.5 s are merged. Turns shorter than 0.4 s are then dropped, because the ASR model invents text on near-silence.
4. Each turn is transcribed by `CohereLabs/cohere-transcribe-arabic-07-2026` with `language="ar"` fixed (the model has no language detection). The model loads once. Device order is CUDA, then MPS, then CPU.
5. Roles: the speaker with the highest RMS loudness (dBFS over all their kept turns) is `primary`, on the assumption that they are closest to the mic. Everyone else is `other`. Loudness per speaker is printed so you can check this. `--primary SPEAKER_01` overrides it.
6. Output goes to `out/<input name>/`.

## Install

Needs macOS or Linux, Python 3.12, [uv](https://docs.astral.sh/uv/) and ffmpeg.

```bash
brew install uv ffmpeg          # Linux: install ffmpeg with your package manager, uv via https://astral.sh/uv
git clone https://github.com/osama-alharbi9/arabic-diarized-transcribe
cd arabic-diarized-transcribe
uv sync                         # creates .venv with Python 3.12 from uv.lock
cp .env.example .env            # then set HF_TOKEN (and COHERE_API_KEY if you use --asr api)
```

Instead of putting `HF_TOKEN` in `.env`, you can run `uv run hf auth login` once and approve in the browser. The script falls back to that saved token.

Both models are gated. Accept the terms on both model pages with the account that owns the token:

- https://huggingface.co/pyannote/speaker-diarization-community-1
- https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026

The first local run downloads about 4.1 GB of Cohere weights plus the pyannote pipeline into `~/.cache/huggingface`.
Environment variables already set in your shell take precedence over `.env`.

## Run

```bash
uv run python transcribe.py recording.m4a                        # local ASR, auto device
uv run python transcribe.py recording.m4a --primary SPEAKER_01   # override the loudness heuristic
uv run python transcribe.py recording.m4a --max-speakers 2
uv run python transcribe.py recording.m4a --asr api              # Cohere API instead of local weights
uv run python transcribe.py recording.m4a --device cpu           # force a device
```

Progress, per-speaker loudness and each transcribed turn are printed to stderr.

`--asr api` sends each turn (not the whole file) to `POST https://api.cohere.com/v2/audio/transcriptions` with `model=cohere-transcribe-arabic-07-2026` and `language=ar`.
Diarization still runs locally, which works on CPU.
The free tier is rate-limited, so the script retries on HTTP 429 and 5xx with backoff.
Each turn has to be under the API's 25 MB limit. At 16 kHz 16-bit mono that is about 13 minutes per turn.

## Output

`out/<name>/transcript.json`: a list of `{start, end, speaker, role, text}`, with times in seconds.

`out/<name>/transcript.txt`: one line per turn.

```
[00:00] Primary: ...
[00:06] Other: ...
```

`out/<name>/metrics.json`:

| field | meaning |
|---|---|
| `audio_duration_s` | length of the converted audio |
| `speakers_found` | distinct speakers in the exclusive diarization |
| `turns_raw` / `turns_kept` / `turns_dropped` | turns before cleanup, after cleanup, and dropped for being under 0.4 s |
| `diarization_s` | diarization wall time, including pipeline load |
| `asr_s` | wall time spent transcribing turns, excluding model load |
| `asr_model_load_s` | Cohere model load time (local), near 0 for api |
| `rtfx` | `audio_duration_s / (diarization_s + asr_s)`. Higher is faster. |
| `device` | torch device used for diarization and local ASR |
| `primary` / `primary_source` | which speaker got the `primary` role, and whether that came from loudness or `--primary` |
| `speaker_loudness` | per speaker: RMS dBFS, seconds of speech, turns. This drives the role heuristic. |

## What was verified and what was not

Tested in October 2026 on macOS 26, Apple M4 with 24 GB RAM, Python 3.12.13, uv 0.13.5, ffmpeg 9.0.2.
Test audio was a 78 s synthetic two-speaker dialogue (macOS TTS reading a Najdi script, second voice pitch-shifted and attenuated) and a 60 s Arabic podcast clip. Neither is included in this repo.

**Verified**

| item | result |
|---|---|
| One venv, no torch conflict | `uv.lock` resolves pyannote.audio 4.0.7 + transformers 5.20.2 on torch 2.14.1 (torchaudio 2.11.0, torchcodec 0.17.0). No split needed. |
| Diarization on MPS | Works. 10.3 s for 78.5 s of audio. CPU takes 40.6 s. |
| Cohere ASR on MPS, bfloat16 | Works. 7.3 s for 15 turns. |
| Cohere ASR on CPU, float32 | Works. 21.1 s for the same 15 turns. Text matches MPS except one letter in one turn. |
| Whole pipeline (`rtfx`) | 4.46 (synthetic) and 3.16 (podcast) on MPS, 1.27 on CPU |
| Inputs | m4a and mp3 end to end. Stereo mp3 (44.1 kHz) and wav (48 kHz) convert to 16 kHz mono. |
| Flags | `--primary`, `--max-speakers`, `--device cpu`, plus errors for an unknown speaker, a missing key and a missing file |
| Loudness heuristic | Synthetic dialogue: correct, −20.1 vs −30.9 dBFS. Podcast: wrong, because both speakers had close mics and were only 1.3 dB apart. |

**Not verified**

| item | why |
|---|---|
| Accuracy on real field recordings | Only the synthetic dialogue and one podcast clip were tested. |
| `--asr api` | No `COHERE_API_KEY` was available. The endpoint exists (it returns 401 without a key). The model id `cohere-transcribe-arabic-07-2026` comes from Cohere's docs. Request handling and the 429 retry have never run. |
| CUDA | No NVIDIA GPU was available. The code picks bfloat16 on CUDA. |
| Linux | Only macOS was tested. |
| Version bounds | `pyproject.toml` sets lower bounds only. `uv.lock` pins the exact versions that were tested, so use `uv sync` rather than `pip install`. |

## Known limits

- **Diarization is the weakest part.** On the podcast clip, community-1 missed a speaker change after a 0.3 s pause, folded one-word replies into the other speaker's turn, and split a sentence at the wrong point. pyannote's paid `precision-2` model is the closest upgrade (a one-line change), but it hasn't been tested here.
- **Overlap is lost.** Exclusive diarization gives each moment to one speaker, so crosstalk is either dropped or transcribed inside the other speaker's turn.
- **Code-switching is partial.** In a spliced test, English words usually came out in Latin script, often wrapped in `( )` or `« »`. Some were dropped, such as "warranty" and "gig". Common loanwords (email, WhatsApp) came out in Arabic script. The model card says performance on code-switched audio is "inconsistent".
- **Turn edges are cut exactly at diarization boundaries,** with no padding. A word clipped at a boundary can be lost or misheard.
- **Cleanup merges first, then drops.** A short blip from another speaker inside a turn blocks the merge, then gets dropped, and leaves the turn split in two. In testing, one such 0.8 s fragment was transcribed as an unrelated word. Running the merge again after the drop would fix this.
- **On near-silence the model outputs placeholder text** such as `@@@فراغ`. The 0.4 s minimum turn length keeps most of this out.
- **The loudness heuristic assumes** the primary speaker is clearly closest to the mic and that diarization didn't merge two people into one speaker. Check the printed dBFS gap. A gap of only 1 or 2 dB means the heuristic is guessing.
- Turns over 35 s are split by the Cohere feature extractor at energy-based points and stitched back together. That is the model's own long-form path.
