# Depth Map Video Benchmark

Measured: 2026-10-31

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB |
| Driver | 621.01 |
| PyTorch | 2.11.0+cu128 |
| Precision | CUDA FP16 |
| Input | `bench1080.mp4`, 1920x1080, 30 fps, 90 frames |
| Preset | Fast, 392-pixel model input |
| Encoder | `auto`, CRF 18 |

Model loading is measured separately and excluded from render time. Each render
includes decode, depth inference, grayscale conversion, and encode.

## Results

| Metric | Result |
| --- | ---: |
| Model load | 0.532 s |
| 720p render, 1280x720 | 5.970 s, 15.08 fps |
| 1080p render, 1920x1080 | 8.712 s, 10.33 fps |

The earlier Turbo/336 run measured 17.76 fps at 720p and 11.07 fps at 1080p.
Fast/392 is slower, but it completed both benchmark renders without a hard
failure. The approved Fast default therefore remains unchanged.

## Temporal Stability

A 10-frame comparison on `bench1080.mp4` measured the mean absolute
frame-to-frame difference inside static background pixels:

| Mode | Mean frame-to-frame difference |
| --- | ---: |
| Temporal stability off | 5.95 / 255 |
| Temporal stability on | 1.36 / 255 |

The motion-masked temporal blend reduced static-area flicker by approximately
77% without optical flow or a measurable render-time penalty.

Benchmark outputs and source clips stay under the ignored `out/` directory.
This file is the tracked benchmark record.
