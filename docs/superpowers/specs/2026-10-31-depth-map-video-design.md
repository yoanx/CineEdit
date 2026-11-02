# CineEdit Depth Map Video Design

Date: 2026-10-31
Status: Approved for specification review

## Goal

Add a second processing mode that generates a grayscale monocular depth-map video for AE/PR compositing. The existing motion-blur mode remains unchanged. Users switch modes from a new leftmost rail while keeping the current source video loaded.

The default output is an 8-bit H.264 MP4 with no audio, near objects rendered white, far objects rendered black, and the same frame rate and trim range as the source.

## Non-Goals

- Point-cloud or 3D reconstruction output
- EXR, 16-bit, or lossless depth intermediates
- Stereo depth
- Optical-flow temporal stabilization
- Audio in depth-map videos

## Architecture

### New Processing Components

- `DepthEngine`
  - Loads `depth-anything/Depth-Anything-V2-Small-hf`.
  - Uses CUDA FP16 when available.
  - Keeps the model resident between previews and full renders.
  - Converts model relative depth into normalized grayscale depth.
- `DepthRenderer`
  - Runs decode, depth estimation, grayscale conversion, and encode.
  - Reuses the existing ffmpeg decoder/encoder and progress/cancel contracts.
  - Does not load RAFT or the motion-blur pipeline.
- `DepthConfig`
  - Contains speed preset, output height, invert flag, codec, trim range, and device.

### Existing Components

- `webui.py` gains a job mode and depth preview branch.
- `ui.html` gains mode state, the left rail, depth-specific inspector controls, and depth view labels.
- `CineEdit.spec` includes the packaged depth model and required Python data files.
- `requirements.txt` adds a pinned Transformers release compatible with the model and PyInstaller.

## Model And Depth Processing

- Default speed preset: 392-pixel model input, FP16, batch size 1.
- Turbo preset: 336-pixel model input.
- Fine preset: 518-pixel model input.
- Model input dimensions must remain multiples of 14.
- The source frame is resized for inference, then the predicted depth is resized to the requested output resolution.
- Depth normalization uses per-frame 1st/99th percentiles. The normalized depth is then blended with the previous stable frame only where the RGB frame difference is below `0.02`.
- The temporal blend uses a previous-frame weight of `0.75`. If fewer than 15% of pixels are static, the frame is treated as a camera/scene transition and the current depth is kept.
- Default mapping: higher inverse depth becomes white; lower inverse depth becomes black.
- Invert mode swaps black and white.
- No optical-flow temporal stabilization is used; the temporal blend is motion-masked and does not require RAFT.

## Render Pipeline

1. Probe the source and resolve output dimensions.
2. Decode RGB frames with the existing reader thread.
3. Infer relative depth on GPU.
4. Normalize and convert to a single-channel grayscale frame.
5. Replicate grayscale to RGB24 for the existing encoder.
6. Encode H.264 with NVENC when available, otherwise libx264.
7. Return frame count, elapsed time, fps, output path, and profile timings.

The output path defaults to `<source-stem>_depth.mp4`. Audio is never mapped.

## Preview

- Endpoint: `/api/frame?mode=depth`.
- The preview uses the same timeline and compare-wipe behavior as motion blur.
- Preview view labels become `Compare / Original / Depth` and their Chinese equivalents.
- The original frame is cached independently of depth parameters.
- Depth maps are cached by source, timestamp, quality preset, invert flag, and preview height.
- First entry into depth mode may show a model-loading state; subsequent previews reuse the loaded model.

## UI

### Left Rail

- Desktop width: 72 px.
- Narrow width: 56 px with labels hidden and accessible names retained.
- Two entries: Motion Blur and Depth Map.
- Linear monochrome SVG icons, active state expressed by background block and font weight.
- Keyboard focus order follows the visual order.

### Mode Switching

- Switching modes keeps the current source video.
- Motion-blur and depth parameters are stored independently.
- Each mode keeps its own output path.
- Switching modes updates the stage labels, inspector, preview cache key, output suffix, and render action without reloading the source.
- The selected mode is persisted in local storage.

### Depth Inspector

- Speed preset: Turbo, Fast, Fine; Fast is default.
- Output resolution: Source, 1080p, 720p.
- Invert depth toggle.
- Encoder: Auto, H.264 NVENC, libx264.
- Trim range and output path.
- Render progress, cancel, elapsed time, fps, and profile breakdown.

### Localization

- All new UI strings support Chinese and English.
- File names and paths remain untranslated.
- Language and mode state persist independently.

## Performance

- Depth model loading occurs outside measured render time.
- FP16 and `torch.inference_mode()` are required on CUDA.
- Decode and encode remain in separate threads from GPU inference.
- If CUDA memory is insufficient, retry once at the next lower input preset.
- The implementation must benchmark preview latency and full-render fps at 720p and 1080p on the current RTX 5060 Laptop, then lower the default input preset if Fast does not meet the agreed speed-first priority.

## Error Handling

- Missing model files produce a clear bilingual error and do not fall back to network download during normal startup.
- Missing ffmpeg continues to use the existing error path.
- Unsupported video extensions remain rejected at the upload boundary.
- Cancel requests stop after the current frame and still close decoder/encoder processes.
- Encoder failure includes the last portion of ffmpeg stderr.

## Packaging

- The Depth Anything V2 Small weights are copied under `assets/weights/depth_anything_v2_small`.
- PyInstaller includes the model files, Transformers package data, and tokenizer/configuration files.
- The installed application must work without network access.
- Expected installer growth is in the hundreds of megabytes.

## Testing

- Unit tests for depth normalization, inversion, motion-masked temporal blending, and output shape.
- Pipeline test with a short real video: output frame count, no audio stream, grayscale content, and successful cancellation.
- Real-model smoke test on one image and one short video.
- API tests for depth preview and job mode.
- Playwright checks for left-rail switching, state retention, bilingual labels, timeline scrubbing, and responsive layouts.
- Offline packaging test with networking disabled.
- Performance benchmark recorded separately from model-load time.

## Acceptance Criteria

- Motion blur remains functionally unchanged.
- Switching to Depth Map and back preserves the source and both modes' settings.
- Depth preview shows the complete depth frame in Depth view.
- A one-second 720p clip renders a playable grayscale depth video with no audio.
- The output frame rate and trim range match the source.
- The UI has no horizontal overflow at 390, 1024, and 1440 pixels.
- The packaged application loads the depth model and runs offline.
