# Hallo-Live Streaming Project Context

## Project
Repository: https://github.com/fudan-generative-vision/Hallo-Live
Path: /workspace/Hallo-Live
Virtual environment: /workspace/Hallo-Live/hallo-venv

Hardware:
- RunPod RTX PRO 6000 Blackwell Server Edition
- Approximately 96 GB GPU VRAM
- Existing inference installation works
- Preserve existing dependencies and model files

## Verified inference results
Video:
- 992x512
- 24 FPS
- 117 frames
- Duration 4.875 seconds

Audio:
- WAV PCM 16-bit
- 16000 Hz mono
- Duration 4.806563 seconds

Performance:
- Pipeline inference approximately 16.63 seconds
- Diffusion generation approximately 6.60 seconds
- VAE decoding approximately 9.38 seconds
- Generated video duration 4.875 seconds
- Not currently real-time

## Identified problems
1. Speech sometimes drops the final two words.
2. Model has a character limit; inspect the source to determine its exact value.
3. The previous offline inference test does not prove live streaming.
4. Internal six-frame KV blocks do not automatically mean chunks are delivered to the browser.
5. Need to verify whether audio and video are generated and decoded incrementally.
6. Avoid GPU OOM and multiple model instances.

## Required application
Implement:
- Persistent Hallo-Live inference worker
- Automatic sentence-aware text chunking
- Strict enforcement of model character limit
- No lost text during chunking
- Audio and video generation
- Backend streaming API
- Browser demo with synchronized playback
- Generation progress and performance metrics
- Time to first audio
- Time to first video
- Streaming generation FPS
- Error handling and logging

## Existing prototype
A stream_demo bundle was previously generated in ChatGPT.
Inspect stream_demo/ if present, but do not assume its code is correct.
It uses completed MP4 segments rather than confirmed internal block streaming.

## Important engineering requirements
1. Inspect actual repository implementation before editing.
2. Identify model prompt length handling.
3. Identify text encoder, inference loop, KV cache and VAE decoding.
4. Preserve the loaded model across requests.
5. Avoid reloading weights for every text chunk.
6. Verify whether model state can continue across text chunks.
7. Avoid claiming true real-time streaming without measurements.
8. Verify the final words in generated speech.
9. Provide complete working code and commands.
10. Test against the installed environment rather than inventing APIs.
EOF

cat > AGENTS.md <<'EOF'
# Hallo-Live Project Instructions

Read CODEX_CONTEXT.md before implementing changes.

Inspect the existing repository, model configuration and inference
implementation. Treat the repository as the source of truth.

Use /workspace/Hallo-Live/hallo-venv.

Do not reinstall PyTorch, CUDA or model dependencies unnecessarily.
Do not overwrite model checkpoints.
Do not launch multiple GPU inference workers.

Implement changes incrementally and test each stage.

Distinguish completed-segment delivery from true incremental
audio/video block streaming.

Document changes, test results, unresolved problems and next steps.
