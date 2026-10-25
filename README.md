# Hallo-Live chunked streaming demo

This is a persistent-model, segment-level streaming demo:
- The model is initialized once by the FastAPI process.
- A long spoken script is split into exact contiguous chunks.
- Every *complete prompt* is checked against `HALLOLIVE_MAX_PROMPT_CHARS`.
- Each generated segment is saved as an MP4 with synchronized audio/video.
- The browser starts playing the first segment while later segments are generated.
- Metrics include time to first playable segment, frames, media duration, segment FPS, and cumulative FPS.

Important: the upstream `pipeline.inference()` returns decoded output only after its generation loop finishes. This demo therefore streams **completed text segments**, not each internal six-frame causal block. It is a reliable first demo without modifying the upstream inference implementation. True per-block delivery is a separate pipeline patch.

## 1. Configure exact paths and limit

Set the exact character limit you found in the model code. Do not use a guessed value.

```bash
cd /workspace/Hallo-Live
source hallo-venv/bin/activate

export HALLOLIVE_MODEL_DIR="/absolute/path/to/your/model/root"
export HALLOLIVE_CHECKPOINT="$HALLOLIVE_MODEL_DIR/Hallo-Live/hallolive_dit.pt"
export HALLOLIVE_CONFIG="/workspace/Hallo-Live/configs/dual_stream_dmd_5B.yaml"
export HALLOLIVE_MAX_PROMPT_CHARS="REPLACE_WITH_THE_VERIFIED_INTEGER"
```

If the checkpoint is elsewhere, set `HALLOLIVE_CHECKPOINT` to its exact path. The model root must contain the text encoder, video VAE and audio VAE assets expected by the existing installation.

## 2. Install web dependencies in the existing venv

```bash
cd /workspace/Hallo-Live
source hallo-venv/bin/activate
python -m pip install fastapi "uvicorn[standard]"
```

This does not reinstall Torch or the model stack.

## 3. Start the service

```bash
cd /workspace/Hallo-Live
source hallo-venv/bin/activate
export HALLOLIVE_MODEL_DIR="/absolute/path/to/your/model/root"
export HALLOLIVE_CHECKPOINT="$HALLOLIVE_MODEL_DIR/Hallo-Live/hallolive_dit.pt"
export HALLOLIVE_CONFIG="/workspace/Hallo-Live/configs/dual_stream_dmd_5B.yaml"
export HALLOLIVE_MAX_PROMPT_CHARS="REPLACE_WITH_THE_VERIFIED_INTEGER"
python -m uvicorn stream_demo.server:app --host 0.0.0.0 --port 8090
```

Open `http://YOUR_RUNPOD_HOST_OR_FORWARDED_URL:8090/`. For a RunPod HTTP service, expose/forward port 8090 using your pod's normal port mapping.

## Metrics interpretation

- **Time to first playable audio/video**: from the browser job start to the first complete segment becoming available. This is not first causal-block latency.
- **Segment generation FPS**: output frames reported by ffprobe divided by that segment's inference time.
- **Cumulative FPS**: all completed output frames divided by elapsed time since job start.
- **Text preservation**: the backend checks that the concatenated input chunks equal the original script exactly. This proves no characters were lost during splitting, not that the generative model will pronounce every word. Listen to the last sentence to validate speech content.

## Notes

- Only one generation job should use the single GPU at a time. The UI currently opens one WebSocket job at a time.
- Each text chunk is a new inference call and starts a new causal KV cache; the model is kept loaded, but causal state is not carried across text chunks. Clips are played sequentially; minor visual/voice discontinuity between chunks is possible.
- If the static scene and audio captions alone consume most of the verified prompt budget, shorten those descriptions.
- `ffprobe` and `ffmpeg` should be installed in the pod.
