#!/usr/bin/env python3
"""
Hallo-Live chunked-text streaming demo.
Keeps the model loaded once; generates one bounded prompt segment at a time and
notifies the browser as each completed audio/video MP4 becomes playable.
This is segment-level streaming, not per-causal-block delivery.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
OUTPUT_ROOT = ROOT / "output" / "stream_demo"
CONFIG_PATH = Path(os.environ.get("HALLOLIVE_CONFIG", ROOT / "configs/dual_stream_dmd_5B.yaml"))
MODEL_DIR = Path(os.environ.get("HALLOLIVE_MODEL_DIR", ""))
CHECKPOINT = Path(os.environ.get("HALLOLIVE_CHECKPOINT", ""))
MAX_PROMPT_CHARS = int(os.environ.get("HALLOLIVE_MAX_PROMPT_CHARS", "0"))
FPS = 24
SAMPLE_RATE = 16000

DEFAULT_SCENE = (
    "A realistic human avatar in a clean, softly lit studio, head-and-shoulders framing, "
    "camera fixed, face clearly visible, natural subtle expressions and readable lip movement."
)
DEFAULT_AUDIO = (
    "Clear, natural conversational voice, steady pace, clean close-mic speech, "
    "subtle room tone, no music."
)

app = FastAPI(title="Hallo-Live Chunked Streaming Demo")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_ROOT), check_dir=False), name="outputs")

# Loaded once on startup.
PIPELINE = None
DEVICE = None
CONFIG = None
LOW_MEMORY = False
MODEL_READY = False
MODEL_ERROR = None


def _sample_noise(sample_seeds, shape, device, dtype):
    items = []
    for seed in sample_seeds:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        items.append(torch.randn(shape, generator=gen, device=device, dtype=dtype))
    return torch.stack(items, dim=0)


def _initialize_model():
    global PIPELINE, DEVICE, CONFIG, LOW_MEMORY, MODEL_READY, MODEL_ERROR
    if MAX_PROMPT_CHARS < 1:
        raise RuntimeError(
            "Set HALLOLIVE_MAX_PROMPT_CHARS to the exact character limit you verified in the model code."
        )
    if not MODEL_DIR or not MODEL_DIR.exists():
        raise RuntimeError(
            "Set HALLOLIVE_MODEL_DIR to the existing model root (the directory containing Ovi/Wan/MMAudio assets)."
        )
    if not CHECKPOINT:
        candidate = MODEL_DIR / "Hallo-Live" / "hallolive_dit.pt"
        if candidate.exists():
            ckpt = candidate
        else:
            raise RuntimeError(
                "Set HALLOLIVE_CHECKPOINT to the existing hallolive_dit.pt checkpoint; default path was not found."
            )
    else:
        ckpt = CHECKPOINT
    if not ckpt.is_file():
        raise RuntimeError(f"Checkpoint not found: {ckpt}")
    if not CONFIG_PATH.is_file():
        raise RuntimeError(f"Config not found: {CONFIG_PATH}")
    if not torch.cuda.is_available():
        raise RuntimeError("Hallo-Live inference requires CUDA, but torch.cuda.is_available() is false.")

    import hallolive.utils.memory as memory_utils
    from hallolive.pipeline import dual_stream_causal_inference
    from hallolive.pipeline import DualStreamCausalInferencePipeline
    from hallolive.utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb

    torch.cuda.set_device(0)
    DEVICE = torch.device("cuda:0")
    # Upstream inference.py patches these cached device references after selecting the CUDA device.
    memory_utils.gpu = DEVICE
    dual_stream_causal_inference.gpu = DEVICE
    torch.set_grad_enabled(False)

    default_cfg_path = ROOT / "configs" / "default_config.yaml"
    cfg = OmegaConf.merge(OmegaConf.load(default_cfg_path), OmegaConf.load(CONFIG_PATH))
    cfg.model_dir = str(MODEL_DIR)
    cfg.generator_ckpt = str(ckpt)
    cfg.use_ema = True
    cfg.profile = False
    cfg.i2v = False
    cfg.num_samples = 1
    cfg.seed = int(cfg.seed)
    cfg.num_output_frames = int(cfg.num_output_frames)
    cfg.generator_meta_init = True
    CONFIG = cfg

    print("[stream-demo] Initializing Hallo-Live model...", flush=True)
    t0 = time.perf_counter()
    PIPELINE = DualStreamCausalInferencePipeline(cfg, device=DEVICE)
    print(f"[stream-demo] Pipeline init: {time.perf_counter()-t0:.2f}s", flush=True)

    t0 = time.perf_counter()
    state = torch.load(str(ckpt), map_location="cpu", mmap=True, weights_only=True)
    if cfg.use_ema and isinstance(state, dict) and "generator_ema" in state:
        state = state["generator_ema"]
        print("[stream-demo] Loading EMA generator weights", flush=True)
    elif isinstance(state, dict) and "generator" in state:
        state = state["generator"]
        print("[stream-demo] Loading generator weights", flush=True)
    state = {k.replace("_fsdp_wrapped_module.", ""): v for k, v in state.items()}
    PIPELINE.generator.load_state_dict(state, assign=bool(cfg.generator_meta_init))
    print(f"[stream-demo] Checkpoint load: {time.perf_counter()-t0:.2f}s", flush=True)

    free_gb = get_cuda_free_memory_gb(DEVICE)
    LOW_MEMORY = free_gb < 40
    t0 = time.perf_counter()
    if LOW_MEMORY:
        print(f"[stream-demo] Free VRAM {free_gb:.2f} GB; enabling text-encoder dynamic swap", flush=True)
        DynamicSwapInstaller.install_model(PIPELINE.text_encoder, device=DEVICE)
    else:
        PIPELINE.text_encoder.to(device=DEVICE)
    PIPELINE.generator.to(device=DEVICE, dtype=torch.bfloat16)
    if PIPELINE.generator_meta_init:
        PIPELINE.generator.model.set_rope_params()
    PIPELINE.vae_video.to(device=DEVICE, dtype=torch.bfloat16)
    PIPELINE.vae_audio.to(device=DEVICE, dtype=torch.bfloat16)
    print(f"[stream-demo] Move models to GPU: {time.perf_counter()-t0:.2f}s", flush=True)
    MODEL_READY = True
    print("[stream-demo] Model ready.", flush=True)


def _split_exact(text: str, max_chars: int) -> list[str]:
    """Split into bounded, contiguous slices without dropping or rewriting characters."""
    if not text:
        return []
    if max_chars < 1:
        raise ValueError("Prompt budget is too small after adding the fixed prompt template.")
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end_limit = min(start + max_chars, n)
        if end_limit == n:
            chunks.append(text[start:n])
            break

        # Prefer a sentence boundary within the budget.
        window = text[start:end_limit]
        sentence_ends = list(re.finditer(r'[.!?]["\')\]]*\s+', window))
        if sentence_ends:
            cut = start + sentence_ends[-1].end()
        else:
            # Otherwise split at whitespace so words are not cut when possible.
            whitespace = list(re.finditer(r'\s+', window))
            if whitespace:
                cut = start + whitespace[-1].end()
            else:
                # A single unbroken token longer than the budget must be split to obey the hard limit.
                cut = end_limit
        if cut <= start:
            cut = end_limit
        chunks.append(text[start:cut])
        start = cut
    assert "".join(chunks) == text, "Internal splitter error: text changed or was lost"
    assert all(len(c) <= max_chars for c in chunks)
    return chunks


def _build_prompt(scene: str, speech: str, audio_caption: str) -> str:
    return f"{scene.strip()} <S>{speech}<E> <AUDCAP>{audio_caption.strip()}<ENDAUDCAP>"


def _probe_mp4(path: Path) -> dict[str, Any]:
    try:
        raw = subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,avg_frame_rate,nb_frames",
                "-show_entries", "format=duration",
                "-of", "json", str(path),
            ],
            text=True, stderr=subprocess.STDOUT, timeout=20,
        )
        data = json.loads(raw)
        stream = (data.get("streams") or [{}])[0]
        duration = float((data.get("format") or {}).get("duration") or 0)
        frames = int(stream.get("nb_frames") or 0)
        if not frames and duration:
            m = re.match(r"(\d+)/(\d+)", str(stream.get("avg_frame_rate", "0/1")))
            rate = float(m.group(1)) / float(m.group(2)) if m and int(m.group(2)) else FPS
            frames = round(duration * rate)
        return {
            "duration_seconds": duration,
            "frames": frames,
            "width": stream.get("width"),
            "height": stream.get("height"),
            "frame_rate": stream.get("avg_frame_rate", "unknown"),
        }
    except Exception as exc:
        return {"duration_seconds": 0, "frames": 0, "probe_warning": str(exc)}


def _generate_segment(prompt: str, output_path: Path) -> dict[str, Any]:
    from inference import get_noise_shapes
    from hallolive.utils.memory import get_cuda_free_memory_gb
    from hallolive.ovi.utils.io_utils import save_video

    # Ensure every complete prompt, including the scene/audio wrappers, obeys the discovered limit.
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(
            f"Internal guard: prompt has {len(prompt)} chars, over configured limit {MAX_PROMPT_CHARS}."
        )

    video_noise_shape, _, audio_noise_shape = get_noise_shapes(CONFIG)
    seed = int(CONFIG.seed)
    video_noise = _sample_noise([seed], video_noise_shape, DEVICE, torch.bfloat16)
    audio_noise = _sample_noise([seed], audio_noise_shape, DEVICE, torch.bfloat16)
    t0 = time.perf_counter()
    with torch.inference_mode():
        video, audio = PIPELINE.inference(
            video_noise=video_noise,
            audio_noise=audio_noise,
            text_prompts=[prompt],
            return_latents=False,
            profile=False,
            low_memory=LOW_MEMORY,
        )
        torch.cuda.synchronize(DEVICE)
    inference_seconds = time.perf_counter() - t0
    video_np = video.detach().cpu().float().numpy()
    audio_np = audio.detach().cpu().float().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t_save = time.perf_counter()
    save_video(str(output_path), video_np[0], audio_np[0].squeeze(), fps=FPS, sample_rate=SAMPLE_RATE)
    save_seconds = time.perf_counter() - t_save
    media = _probe_mp4(output_path)
    total_seconds = time.perf_counter() - t0
    media["inference_seconds"] = round(inference_seconds, 3)
    media["save_seconds"] = round(save_seconds, 3)
    media["generation_fps"] = round(media["frames"] / inference_seconds, 3) if media["frames"] and inference_seconds else None
    media["end_to_end_fps"] = round(media["frames"] / total_seconds, 3) if media["frames"] and total_seconds else None
    media["file_bytes"] = output_path.stat().st_size
    return media


@app.on_event("startup")
async def startup():
    global MODEL_ERROR
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        # Keep model loading off the event loop.
        await asyncio.to_thread(_initialize_model)
    except Exception as exc:
        MODEL_ERROR = f"{type(exc).__name__}: {exc}"
        print(f"[stream-demo] MODEL STARTUP FAILED: {MODEL_ERROR}", flush=True)
        # Let the UI and /health explain the error instead of hiding it in a crash loop.


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    return {
        "ok": MODEL_READY,
        "model_ready": MODEL_READY,
        "model_error": MODEL_ERROR,
        "max_prompt_chars": MAX_PROMPT_CHARS,
        "config": str(CONFIG_PATH),
        "model_dir": str(MODEL_DIR) if str(MODEL_DIR) else "",
        "checkpoint": str(CHECKPOINT) if str(CHECKPOINT) else str(MODEL_DIR / "Hallo-Live" / "hallolive_dit.pt"),
    }


@app.websocket("/ws")
async def stream_ws(ws: WebSocket):
    await ws.accept()
    try:
        request = await ws.receive_json()
        if not MODEL_READY:
            await ws.send_json({"type": "error", "message": f"Model is not ready: {MODEL_ERROR or 'initializing'}"})
            await ws.close(code=1011)
            return

        script = str(request.get("text", ""))
        if not script.strip():
            await ws.send_json({"type": "error", "message": "Please enter the text to speak."})
            await ws.close()
            return
        scene = str(request.get("scene_prompt") or DEFAULT_SCENE)
        caption = str(request.get("audio_caption") or DEFAULT_AUDIO)
        prefix = f"{scene.strip()} <S>"
        suffix = f"<E> <AUDCAP>{caption.strip()}<ENDAUDCAP>"
        available = MAX_PROMPT_CHARS - len(prefix) - len(suffix)
        if available < 1:
            await ws.send_json({
                "type": "error",
                "message": f"The fixed scene/audio prompt uses {len(prefix)+len(suffix)} characters, exceeding the configured limit {MAX_PROMPT_CHARS}. Shorten the scene/audio descriptions or verify the limit.",
            })
            await ws.close()
            return
        chunks = _split_exact(script, available)
        # The concatenation check is a hard guard against dropped words/characters.
        if "".join(chunks) != script:
            raise RuntimeError("Chunking validation failed; text was not preserved exactly.")
        job_id = uuid.uuid4().hex[:12]
        job_dir = OUTPUT_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        total_frames = 0
        total_media_seconds = 0.0
        first_ready_seconds = None
        await ws.send_json({
            "type": "started", "job_id": job_id, "chunks": len(chunks),
            "script_chars": len(script), "max_prompt_chars": MAX_PROMPT_CHARS,
            "max_speech_chars_per_chunk": available,
            "text_preserved_exactly": True,
        })

        for index, speech_chunk in enumerate(chunks, start=1):
            prompt = _build_prompt(scene, speech_chunk, caption)
            if len(prompt) > MAX_PROMPT_CHARS:
                raise RuntimeError(f"Prompt {index} exceeded verified character limit.")
            await ws.send_json({
                "type": "chunk_generating", "index": index, "total": len(chunks),
                "prompt_chars": len(prompt), "speech_chars": len(speech_chunk),
                "text": speech_chunk,
            })
            output_path = job_dir / f"chunk_{index:04d}.mp4"
            metrics = await asyncio.to_thread(_generate_segment, prompt, output_path)
            elapsed = time.perf_counter() - started
            if first_ready_seconds is None:
                first_ready_seconds = elapsed
            total_frames += int(metrics.get("frames") or 0)
            total_media_seconds += float(metrics.get("duration_seconds") or 0)
            metrics.update({
                "type": "chunk_ready",
                "job_id": job_id,
                "index": index,
                "total": len(chunks),
                "url": f"/outputs/{job_id}/{output_path.name}",
                "speech_chars": len(speech_chunk),
                "prompt_chars": len(prompt),
                "time_to_first_audio_video_seconds": round(first_ready_seconds, 3),
                "elapsed_seconds": round(elapsed, 3),
                "cumulative_output_frames": total_frames,
                "cumulative_media_seconds": round(total_media_seconds, 3),
                "cumulative_fps": round(total_frames / elapsed, 3) if elapsed > 0 else None,
                "text_preserved_exactly": True,
            })
            await ws.send_json(metrics)

        elapsed = time.perf_counter() - started
        await ws.send_json({
            "type": "complete", "job_id": job_id,
            "chunks": len(chunks), "elapsed_seconds": round(elapsed, 3),
            "total_frames": total_frames,
            "total_media_seconds": round(total_media_seconds, 3),
            "overall_fps": round(total_frames / elapsed, 3) if elapsed > 0 else None,
            "time_to_first_audio_video_seconds": round(first_ready_seconds or 0, 3),
            "text_preserved_exactly": True,
        })
    except WebSocketDisconnect:
        print("[stream-demo] Browser disconnected.", flush=True)
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass
        print(f"[stream-demo] Request failed: {type(exc).__name__}: {exc}", flush=True)
