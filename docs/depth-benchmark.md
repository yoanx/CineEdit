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

Video Depth Anything Small streaming, measured on a 4.88 s 1080p Valorant
segment with 392 input:

| Metric | Result |
| --- | ---: |
| 1080p render, 480x270 preview | 22.87 s, 12.77 fps |
| 1080p render, full resolution | 35.99 s, 8.11 fps |

The VDA stream keeps temporal state across frames and does not use the old
single-frame Depth Anything model or its motion-masked blend.

Benchmark outputs and source clips stay under the ignored `out/` directory.
This file is the tracked benchmark record.
