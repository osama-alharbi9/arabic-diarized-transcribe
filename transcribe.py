#!/usr/bin/env python3
"""Speaker-labelled transcript of an Arabic conversation recording.

Order is fixed because Cohere Transcribe returns no timestamps and no speakers:
diarize -> cut per-speaker turns -> transcribe each turn -> stitch in time order.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

SR = 16_000
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
ASR_MODEL = "CohereLabs/cohere-transcribe-arabic-07-2026"
ASR_LANGUAGE = "ar"  # the model has no language detection
API_URL = "https://api.cohere.com/v2/audio/transcriptions"
API_MAX_BYTES = 25 * 1024 * 1024
MERGE_GAP_S = 0.5
MIN_TURN_S = 0.4


@dataclass
class Turn:
    start: float
    end: float
    speaker: str
    role: str = ""
    text: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def pick_device(requested: str):
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_wav16k(src: Path, dst: Path) -> None:
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(src), "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(dst)]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit("ffmpeg not found on PATH")
    except subprocess.CalledProcessError as e:
        sys.exit(f"ffmpeg failed to convert {src} (exit {e.returncode})")


def diarize(audio: np.ndarray, device, max_speakers: int, token: str) -> list[Turn]:
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)
    if pipeline is None:
        sys.exit(f"Could not load {DIARIZATION_MODEL}. Check HF_TOKEN and that you accepted the model terms.")
    pipeline.to(device)
    waveform = torch.from_numpy(audio).unsqueeze(0)  # (channel, time)
    output = pipeline({"waveform": waveform, "sample_rate": SR}, max_speakers=max_speakers)
    turns = [Turn(round(seg.start, 3), round(seg.end, 3), spk)
             for seg, spk in output.exclusive_speaker_diarization]
    return sorted(turns, key=lambda t: t.start)


def clean_turns(turns: list[Turn]) -> tuple[list[Turn], int]:
    """Merge same-speaker neighbours with a gap < MERGE_GAP_S, then drop turns < MIN_TURN_S."""
    merged: list[Turn] = []
    for t in turns:
        prev = merged[-1] if merged else None
        if prev and prev.speaker == t.speaker and t.start - prev.end < MERGE_GAP_S:
            prev.end = max(prev.end, t.end)
        else:
            merged.append(Turn(t.start, t.end, t.speaker))
    kept = [t for t in merged if t.duration >= MIN_TURN_S]
    return kept, len(merged) - len(kept)


def slice_audio(audio: np.ndarray, t: Turn) -> np.ndarray:
    return audio[int(t.start * SR):int(t.end * SR)]


def rms_dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return float("-inf")
    return 20 * math.log10(math.sqrt(float(np.mean(x.astype(np.float64) ** 2))) + 1e-12)


def speaker_loudness(audio: np.ndarray, turns: list[Turn]) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for spk in sorted({t.speaker for t in turns}):
        own = [t for t in turns if t.speaker == spk]
        samples = np.concatenate([slice_audio(audio, t) for t in own])
        stats[spk] = {"rms_dbfs": round(rms_dbfs(samples), 2),
                      "speech_s": round(sum(t.duration for t in own), 2),
                      "turns": len(own)}
    return stats


class LocalASR:
    def __init__(self, device, token: str):
        import torch
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        self.torch = torch
        dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
        self.processor = AutoProcessor.from_pretrained(ASR_MODEL, token=token)
        self.model = CohereAsrForConditionalGeneration.from_pretrained(
            ASR_MODEL, token=token, dtype=dtype).to(device).eval()

    def transcribe(self, audio: np.ndarray) -> str:
        inputs = self.processor(audio, sampling_rate=SR, return_tensors="pt", language=ASR_LANGUAGE)
        # Turns longer than the extractor's max_audio_clip_s (35 s) are split; this maps chunks back.
        chunk_index = inputs.pop("audio_chunk_index")
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=256)
        text = self.processor.decode(out, skip_special_tokens=True,
                                     audio_chunk_index=chunk_index, language=ASR_LANGUAGE)[0]
        return text.strip()


class ApiASR:
    def __init__(self, api_key: str):
        import requests

        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {api_key}"

    def transcribe(self, audio: np.ndarray) -> str:
        buf = io.BytesIO()
        sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
        data = buf.getvalue()
        if len(data) > API_MAX_BYTES:
            raise RuntimeError(f"turn is {len(data) / 1e6:.1f} MB, over the API's 25 MB limit")
        for attempt in range(6):
            r = self.session.post(API_URL, timeout=120,
                                  data={"model": ASR_MODEL.split("/")[1], "language": ASR_LANGUAGE},
                                  files={"file": ("turn.wav", data, "audio/wav")})
            if r.status_code == 429 or r.status_code >= 500:
                wait = float(r.headers.get("Retry-After") or 2 ** attempt)
                log(f"  API {r.status_code}, retrying in {wait:.0f}s")
                time.sleep(wait)
                continue
            if not r.ok:
                raise RuntimeError(f"Cohere API {r.status_code}: {r.text[:500]}")
            return r.json()["text"].strip()
        raise RuntimeError("Cohere API kept rate-limiting or failing after 6 attempts")


def mmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def main() -> None:
    load_dotenv(Path(__file__).with_name(".env"))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio", type=Path, help="input recording (m4a, mp3, wav, ...)")
    p.add_argument("--asr", choices=["local", "api"], default="local",
                   help="local = Hugging Face weights on this machine; api = Cohere V2 API (COHERE_API_KEY)")
    p.add_argument("--max-speakers", type=int, default=3)
    p.add_argument("--primary", metavar="SPEAKER_XX",
                   help="override the loudness heuristic for the primary speaker, e.g. SPEAKER_01")
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                   help="torch device for diarization and local ASR (auto = cuda > mps > cpu)")
    p.add_argument("--out", type=Path, default=Path("out"), help="output root (default: out/)")
    args = p.parse_args()

    if not args.audio.is_file():
        sys.exit(f"no such file: {args.audio}")
    from huggingface_hub import get_token

    hf_token = os.environ.get("HF_TOKEN") or get_token()
    if not hf_token:
        sys.exit("No Hugging Face token: set HF_TOKEN (shell or .env) or run `uv run hf auth login`")
    api_key = os.environ.get("COHERE_API_KEY")
    if args.asr == "api" and not api_key:
        sys.exit("--asr api needs COHERE_API_KEY (export it or put it in .env)")

    import torch

    torch.manual_seed(0)  # the Cohere feature extractor applies random dither
    device = pick_device(args.device)
    out_dir = args.out / args.audio.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"device: {device}  asr: {args.asr}  out: {out_dir}")

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio_16k.wav"
        to_wav16k(args.audio, wav)
        audio, sr = sf.read(wav, dtype="float32")
    assert sr == SR and audio.ndim == 1
    duration = len(audio) / SR
    log(f"audio: {duration:.1f}s")

    log("diarizing ...")
    t0 = time.perf_counter()
    raw_turns = diarize(audio, device, args.max_speakers, hf_token)
    diarization_s = time.perf_counter() - t0
    turns, dropped = clean_turns(raw_turns)
    speakers = sorted({t.speaker for t in raw_turns})
    log(f"  {len(speakers)} speakers, {len(raw_turns)} raw turns -> {len(turns)} kept, "
        f"{dropped} dropped (<{MIN_TURN_S}s) in {diarization_s:.1f}s")
    if not turns:
        sys.exit("no speech turns left after cleaning")

    loudness = speaker_loudness(audio, turns)
    log("loudness per speaker (kept turns):")
    for spk, st in sorted(loudness.items(), key=lambda kv: -kv[1]["rms_dbfs"]):
        log(f"  {spk}: {st['rms_dbfs']:7.2f} dBFS  {st['speech_s']:6.1f}s speech  {st['turns']} turns")
    if args.primary:
        if args.primary not in loudness:
            sys.exit(f"--primary {args.primary} not among kept speakers: {', '.join(loudness)}")
        primary = args.primary
        log(f"primary: {primary} (from --primary)")
    else:
        primary = max(loudness, key=lambda s: loudness[s]["rms_dbfs"])  # loudest = closest to the mic
        log(f"primary: {primary} (loudest; override with --primary)")
    for t in turns:
        t.role = "primary" if t.speaker == primary else "other"

    log(f"loading ASR ({args.asr}) ...")
    t0 = time.perf_counter()
    asr = LocalASR(device, hf_token) if args.asr == "local" else ApiASR(api_key)
    asr_load_s = time.perf_counter() - t0
    log(f"transcribing {len(turns)} turns ...")
    t0 = time.perf_counter()
    for i, t in enumerate(turns, 1):
        t.text = asr.transcribe(slice_audio(audio, t))
        log(f"  [{i}/{len(turns)}] {mmss(t.start)} {t.speaker} ({t.duration:.1f}s): {t.text}")
    asr_s = time.perf_counter() - t0

    rows = [{"start": round(t.start, 2), "end": round(t.end, 2), "speaker": t.speaker,
             "role": t.role, "text": t.text} for t in turns]
    (out_dir / "transcript.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "transcript.txt").write_text(
        "".join(f"[{mmss(t.start)}] {t.role.capitalize()}: {t.text}\n" for t in turns))
    metrics = {
        "audio_duration_s": round(duration, 2),
        "speakers_found": len(speakers),
        "turns_raw": len(raw_turns),
        "turns_kept": len(turns),
        "turns_dropped": dropped,
        "diarization_s": round(diarization_s, 2),
        "asr_s": round(asr_s, 2),
        "asr_model_load_s": round(asr_load_s, 2),
        "rtfx": round(duration / (diarization_s + asr_s), 2),
        "device": str(device),
        "asr_backend": args.asr,
        "primary": primary,
        "primary_source": "flag" if args.primary else "loudness",
        "speaker_loudness": loudness,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    log(f"wrote {out_dir}/transcript.json, transcript.txt, metrics.json")


if __name__ == "__main__":
    main()
