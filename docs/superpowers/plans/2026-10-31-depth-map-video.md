# Depth Map Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a speed-first Depth Anything V2 video mode that produces grayscale H.264 depth-map videos for AE/PR compositing.

**Architecture:** Add pure depth normalization utilities, a reusable GPU depth engine, and a separate video renderer that shares the existing ffmpeg I/O and job/progress contracts. Add a leftmost mode rail and a separate depth inspector in the existing single-page UI.

**Tech Stack:** Python 3.13, PyTorch CUDA, Transformers, Depth Anything V2 Small, ffmpeg, pywebview, HTML/CSS/JavaScript, unittest, Tabbit Playwright.

**Spec:** `docs/superpowers/specs/2026-10-31-depth-map-video-design.md`

## Global Constraints

- Default model input is 392 pixels; Turbo is 336; Fine is 518; input dimensions must be multiples of 14.
- Depth output is 8-bit H.264 MP4 with no audio.
- Default mapping is near-white and far-black; invert is optional.
- Use CUDA FP16 when available and `torch.inference_mode()`.
- Do not load RAFT or run optical-flow temporal stabilization in depth mode.
- Keep motion-blur mode behavior unchanged.
- Preserve the current source video and independent mode settings when switching modes.
- All new UI strings support Chinese and English.
- The packaged application must load model files offline.
- Use the existing unittest and Tabbit verification workflow; do not add a second test framework.

---

### Task 1: Add Pure Depth Normalization Utilities

**Files:**
- Create: `cinedit/depth_ops.py`
- Create: `tests/test_depth_ops.py`

**Interfaces:**
- Consumes: `torch.Tensor` depth maps shaped `[1,H,W]`.
- Produces:
  - `normalize_depth(depth: torch.Tensor, low: float, high: float, invert: bool = False) -> torch.Tensor`
  - `depth_range(depth: torch.Tensor, sample_step: int = 8) -> tuple[float, float]`
  - `smooth_range(previous: tuple[float, float] | None, current: tuple[float, float], alpha: float = 0.85) -> tuple[float, float]`

- [ ] **Step 1: Write the failing tests**

```python
import unittest

import torch
from cinedit.depth_ops import depth_range, normalize_depth, smooth_range


class DepthOpsTests(unittest.TestCase):
    def test_depth_range_and_normalization(self):
        depth = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        low, high = depth_range(depth, sample_step=1)
        self.assertEqual((low, high), (1.0, 4.0))
        out = normalize_depth(depth, low, high)
        self.assertEqual(out.shape, depth.shape)
        self.assertTrue(torch.isclose(out.min(), torch.tensor(0.0)))
        self.assertTrue(torch.isclose(out.max(), torch.tensor(1.0)))

    def test_invert_depth_swaps_black_and_white(self):
        depth = torch.tensor([[[0.0, 1.0]]])
        normal = normalize_depth(depth, 0.0, 1.0)
        inverted = normalize_depth(depth, 0.0, 1.0, invert=True)
        self.assertTrue(torch.equal(normal, 1.0 - inverted))

    def test_smooth_range_uses_previous_range(self):
        self.assertEqual(smooth_range((0.0, 10.0), (10.0, 20.0), alpha=0.85),
                         (1.5, 11.5))
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `.venv\Scripts\python.exe -m unittest tests.test_depth_ops -v`

Expected: FAIL because `cinedit.depth_ops` does not exist.

- [ ] **Step 3: Implement the minimal utilities**

```python
from __future__ import annotations

import torch


def depth_range(depth: torch.Tensor, sample_step: int = 8) -> tuple[float, float]:
    sample = depth.detach().float().flatten()[::max(1, sample_step)]
    low = torch.quantile(sample, 0.01).item()
    high = torch.quantile(sample, 0.99).item()
    return float(low), float(high)


def normalize_depth(depth: torch.Tensor, low: float, high: float,
                    invert: bool = False) -> torch.Tensor:
    scale = max(high - low, 1e-6)
    out = ((depth.float() - low) / scale).clamp(0.0, 1.0)
    return 1.0 - out if invert else out


def smooth_range(previous: tuple[float, float] | None,
                 current: tuple[float, float],
                 alpha: float = 0.85) -> tuple[float, float]:
    if previous is None:
        return current
    return (
        alpha * previous[0] + (1.0 - alpha) * current[0],
        alpha * previous[1] + (1.0 - alpha) * current[1],
    )
```

- [ ] **Step 4: Run the tests and verify pass**

Run: `.venv\Scripts\python.exe -m unittest tests.test_depth_ops -v`

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```powershell
git add cinedit/depth_ops.py tests/test_depth_ops.py
git commit -m "feat: add depth normalization utilities"
```

---

### Task 2: Add Depth Configuration

**Files:**
- Modify: `cinedit/config.py`
- Create: `tests/test_depth_config.py`

**Interfaces:**
- Consumes: `DepthConfig` quality names `turbo`, `fast`, `fine`.
- Produces:
  - `DepthConfig.input_size() -> int`
  - `DepthConfig.validate() -> None`
  - `DepthConfig.output_name(src: str) -> str`

- [ ] **Step 1: Write the failing tests**

```python
import unittest

from cinedit.config import DepthConfig


class DepthConfigTests(unittest.TestCase):
    def test_depth_input_sizes_are_model_multiples(self):
        self.assertEqual(DepthConfig(quality="turbo").input_size(), 336)
        self.assertEqual(DepthConfig(quality="fast").input_size(), 392)
        self.assertEqual(DepthConfig(quality="fine").input_size(), 518)
        self.assertTrue(all(DepthConfig(quality=q).input_size() % 14 == 0
                            for q in ("turbo", "fast", "fine")))

    def test_depth_output_name_uses_depth_suffix(self):
        self.assertTrue(DepthConfig().output_name("clip.mp4")
                        .endswith("clip_depth.mp4"))
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `.venv\Scripts\python.exe -m unittest tests.test_depth_config -v`

Expected: FAIL because `DepthConfig` is missing.

- [ ] **Step 3: Add `DepthConfig` to `cinedit/config.py`**

```python
@dataclass
class DepthConfig:
    quality: str = "fast"
    output_height: int = 0
    invert: bool = False
    codec: str = "auto"
    crf: int = 18
    device: str = "auto"
    trim_start: float = 0.0
    trim_end: float = 0.0

    def input_size(self) -> int:
        return {"turbo": 336, "fast": 392, "fine": 518}[self.quality]

    def output_name(self, src: str) -> str:
        stem, _ = os.path.splitext(src)
        return stem + "_depth.mp4"

    def validate(self) -> None:
        if self.quality not in ("turbo", "fast", "fine"):
            raise ValueError("quality must be turbo, fast, or fine")
        if self.codec not in ("auto", "h264_nvenc", "libx264"):
            raise ValueError("unsupported depth codec")
```

Add `import os` to `config.py` if not already present.

- [ ] **Step 4: Run the tests and verify pass**

Run: `.venv\Scripts\python.exe -m unittest tests.test_depth_config -v`

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```powershell
git add cinedit/config.py tests/test_depth_config.py
git commit -m "feat: add depth rendering config"
```

---

### Task 3: Add Depth Anything Engine And Offline Weights

**Files:**
- Create: `cinedit/depth_engine.py`
- Create: `assets/weights/depth_anything_v2_small/` with model files
- Modify: `cinedit/weights.py`
- Modify: `requirements.txt`
- Create: `tests/test_depth_engine.py`

**Interfaces:**
- Consumes: RGB tensor `[1,3,H,W]` in `0..1`.
- Produces:
  - `depth_model_dir() -> str`
  - `DepthEngine(model_dir: str | None = None, device: str = "auto")`
  - `DepthEngine.estimate(frame: torch.Tensor, input_size: int) -> torch.Tensor`
  - `DepthEngine.release() -> None`

- [ ] **Step 1: Install the pinned Transformers dependency**

Run:

```powershell
.venv\Scripts\python.exe -m pip install "transformers==4.57.1"
```

Expected: `Successfully installed transformers-4.57.1` or the dependency is already satisfied.

- [ ] **Step 2: Download the model into the project asset directory**

Run:

```powershell
.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('depth-anything/Depth-Anything-V2-Small-hf', local_dir='assets/weights/depth_anything_v2_small')"
```

Expected: `config.json`, `model.safetensors`, and processor configuration files exist under `assets/weights/depth_anything_v2_small`.

- [ ] **Step 3: Write the failing model smoke test**

```python
import unittest

import torch
from cinedit.depth_engine import DepthEngine


class DepthEngineTests(unittest.TestCase):
    def test_depth_engine_returns_finite_full_resolution_depth(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for the depth smoke test")
        engine = DepthEngine(device="cuda")
        frame = torch.rand(1, 3, 128, 160, device=engine.device)
        depth = engine.estimate(frame, input_size=336)
        self.assertEqual(tuple(depth.shape), (1, 1, 128, 160))
        self.assertTrue(torch.isfinite(depth).all())
        engine.release()
```

- [ ] **Step 4: Run the test and verify failure**

Run: `.venv\Scripts\python.exe -m unittest tests.test_depth_engine -v`

Expected: FAIL because `DepthEngine` is missing.

- [ ] **Step 5: Implement offline weight lookup and the engine**

Add to `weights.py`:

```python
DEPTH_MODEL_DIR = "depth_anything_v2_small"

def depth_model_dir() -> str:
    src = _bundle_dir()
    if not src:
        raise FileNotFoundError("depth model directory not found")
    path = os.path.join(src, DEPTH_MODEL_DIR)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"depth model not found: {path}")
    return path
```

Implement `DepthEngine` with `AutoModelForDepthEstimation` and `AutoImageProcessor`, `local_files_only=True`, `torch.inference_mode()`, CUDA FP16, and bilinear upsampling of `predicted_depth` to the input frame dimensions.

- [ ] **Step 6: Add the dependency and run the smoke test**

Add to `requirements.txt`:

```text
transformers==4.57.1
```

Run: `.venv\Scripts\python.exe -m unittest tests.test_depth_engine -v`

Expected: PASS on CUDA; skip only if CUDA is unavailable.

- [ ] **Step 7: Commit**

```powershell
git add cinedit/depth_engine.py cinedit/weights.py requirements.txt assets/weights/depth_anything_v2_small tests/test_depth_engine.py
git commit -m "feat: add offline depth anything engine"
```

---

### Task 4: Add Depth Video Renderer

**Files:**
- Create: `cinedit/depth_pipeline.py`
- Create: `tests/test_depth_pipeline.py`
- Modify: `cinedit/pipeline.py` only if shared helpers are extracted; keep motion-blur behavior unchanged.

**Interfaces:**
- Consumes: `DepthConfig`, `DepthEngine`, `Decoder`, `Encoder`.
- Produces:
  - `DepthRenderer(cfg: DepthConfig, engine: DepthEngine | None = None)`
  - `DepthRenderer.render(src: str, dst: str, progress=None, cancel=None, log=None) -> dict`
  - `DepthRenderer.close() -> None`

- [ ] **Step 1: Write a failing renderer test with a deterministic engine**

```python
import os
import tempfile
import unittest

import torch

from cinedit.config import DepthConfig
from cinedit.depth_pipeline import DepthRenderer
from cinedit.media import probe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "out", "demo_src.mp4")


class ConstantEngine:
    device = torch.device("cpu")

    def estimate(self, frame, input_size):
        return torch.linspace(0.0, 1.0, frame.shape[-2] * frame.shape[-1],
                              device=frame.device).reshape(
                                  1, 1, frame.shape[-2], frame.shape[-1])

    def release(self):
        pass


class DepthPipelineTests(unittest.TestCase):
    def test_depth_renderer_outputs_video_without_audio(self):
        cfg = DepthConfig(quality="fast", device="cpu", codec="libx264",
                          trim_end=0.2)
        with tempfile.TemporaryDirectory() as tmp:
            dst = os.path.join(tmp, "depth.mp4")
            renderer = DepthRenderer(cfg, engine=ConstantEngine())
            stats = renderer.render(DEMO, dst)
            self.assertGreater(stats["frames"], 0)
            self.assertTrue(os.path.isfile(dst))
            info = probe(dst)
            self.assertFalse(info.has_audio)
```

- [ ] **Step 2: Run the test and verify failure**

Run: `.venv\Scripts\python.exe -m unittest tests.test_depth_pipeline -v`

Expected: FAIL because `DepthRenderer` is missing.

- [ ] **Step 3: Implement the renderer**

Use the existing reader/writer thread pattern. Each decoded frame:

```python
frame_t = _to_tensor(frame, engine.device)
depth = engine.estimate(frame_t, cfg.input_size())
low, high = depth_range(depth)
low, high = smooth_range(previous_range, (low, high))
previous_range = (low, high)
gray = normalize_depth(depth, low, high, cfg.invert)
gray_u8 = (gray.clamp(0, 1)[0].mul(255).round().to(torch.uint8).cpu().numpy())
rgb = np.repeat(gray_u8[..., None], 3, axis=2)
enc.write(rgb)
```

Use `audio_from=None` for `Encoder`. Return stats with `frames`, `seconds`, `fps`, `output`, `size`, and `profile`.

- [ ] **Step 4: Run the renderer test**

Run: `.venv\Scripts\python.exe -m unittest tests.test_depth_pipeline -v`

Expected: PASS; output has no audio.

- [ ] **Step 5: Commit**

```powershell
git add cinedit/depth_pipeline.py tests/test_depth_pipeline.py
git commit -m "feat: render depth map videos"
```

---

### Task 5: Add Depth Preview And API Mode Routing

**Files:**
- Modify: `cinedit/preview.py`
- Modify: `cinedit/webui.py`
- Create: `tests/test_depth_webui.py`

**Interfaces:**
- Consumes: query/body field `mode` with values `blur` or `depth`.
- Produces:
  - `build_depth_cfg(payload: dict) -> DepthConfig`
  - `default_output(src: str, mode: str) -> str`
  - `render_depth_preview(src: str, t: float, cfg: DepthConfig, max_height: int = 540) -> np.ndarray`
  - `/api/frame?mode=depth`
  - `/api/start` with `mode=depth`

- [ ] **Step 1: Write failing API/config tests**

```python
import unittest

from cinedit.webui import build_depth_cfg, default_output


class DepthWebUITests(unittest.TestCase):
    def test_build_depth_cfg_maps_speed_preset(self):
        cfg = build_depth_cfg({"mode": "depth", "quality": "turbo",
                               "invert": "1", "preview": "720"})
        self.assertEqual(cfg.quality, "turbo")
        self.assertEqual(cfg.input_size(), 336)
        self.assertTrue(cfg.invert)
        self.assertEqual(cfg.output_height, 720)

    def test_default_depth_output_is_depth_suffix(self):
        self.assertTrue(default_output("clip.mp4", "depth")
                        .endswith("clip_depth.mp4"))
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `.venv\Scripts\python.exe -m unittest tests.test_depth_webui -v`

Expected: FAIL because depth config/routing is missing.

- [ ] **Step 3: Implement mode-aware job routing**

In `start_job`, select:

```python
mode = payload.get("mode", "blur")
if mode == "depth":
    cfg = build_depth_cfg(payload)
    renderer = DepthRenderer(cfg)
else:
    cfg = build_cfg(payload)
    renderer = MotionBlurRenderer(cfg)
```

Implement `default_output(src, mode)` so depth mode returns `<source-stem>_depth.mp4` and blur mode keeps its existing behavior. Use it when `dst` is empty. In `_frame`, branch on `g("mode", "blur")` and call `render_depth_preview`. Reuse `_PREVIEW_LOCK`.

- [ ] **Step 4: Add depth preview cache**

In `preview.py`, cache decoded frames by `(src, mtime, frame_idx, width, height)` and depth output by `(src, mtime, frame_idx, quality, invert, max_height)`. Do not share the motion-blur flow cache.

- [ ] **Step 5: Run tests and verify pass**

Run: `.venv\Scripts\python.exe -m unittest tests.test_depth_webui -v`

Expected: 2 tests pass.

- [ ] **Step 6: Commit**

```powershell
git add cinedit/preview.py cinedit/webui.py tests/test_depth_webui.py
git commit -m "feat: add depth preview and job routing"
```

---

### Task 6: Add Left Mode Rail And Depth Inspector

**Files:**
- Modify: `cinedit/ui.html`
- Modify: `README.md`

**Interfaces:**
- Consumes: API mode `depth` and depth preview response.
- Produces:
  - `S.processMode` with `blur` or `depth`
  - `setProcessMode(mode)` and persisted `cinedit.mode`
  - separate blur/depth parameter panels
  - mode-specific view labels and output suffix

- [ ] **Step 1: Write the failing browser acceptance check**

Run a Tabbit program against the source server:

```javascript
await page.goto("http://127.0.0.1:8863/?open=E%3A%5Cworkbuddy%5C2026-10-27-12-52-57%5Ccinedit%5Cout%5Cdemo_src.mp4");
await expect(page.getByRole("button", {name: "深度图"})).toBeVisible();
await page.getByRole("button", {name: "深度图"}).click();
await expect(page.locator("#depthPanel")).toBeVisible();
await expect(page.locator("#blurPanel")).toBeHidden();
await expect(page.getByRole("button", {name: "深度"})).toBeVisible();
```

Expected before implementation: FAIL because the rail and depth panel do not exist.

- [ ] **Step 2: Add the left rail markup and icons**

Add two buttons with accessible names and SVG icons:

```html
<nav class="mode-rail" aria-label="Processing mode">
  <button id="modeBlur" class="mode-btn on" aria-pressed="true">运动模糊</button>
  <button id="modeDepth" class="mode-btn" aria-pressed="false">深度图</button>
</nav>
```

Use the existing linear SVG icon family. Desktop rail width is 72px; at `max-width: 980px` it is 56px and labels are visually hidden.

- [ ] **Step 3: Add mode state and panel switching**

```javascript
const PROCESS_STORAGE = "cinedit.mode";
S.processMode = localStorage.getItem(PROCESS_STORAGE) || "blur";

function setProcessMode(mode){
  S.processMode = mode === "depth" ? "depth" : "blur";
  localStorage.setItem(PROCESS_STORAGE, S.processMode);
  $("blurPanel").hidden = S.processMode !== "blur";
  $("depthPanel").hidden = S.processMode !== "depth";
  setMode("compare");
  queuePreview(0);
}
```

Keep separate depth inputs: `depthQuality`, `depthRes`, `depthInvert`, `depthCodec`, `depthDst`.

- [ ] **Step 4: Add depth view labels and localization**

Add Chinese and English strings for:

- Motion blur
- Depth map
- Depth view
- Original
- Compare
- Depth quality
- Invert depth
- Depth output path

The top preview segmented control should read `对比 / 原片 / 深度` in depth mode.

- [ ] **Step 5: Preserve independent mode settings**

When switching modes:

- keep `S.src`, duration, fps, timeline position, and trim range
- keep blur inputs untouched
- keep depth inputs untouched
- update the default destination to `_depth.mp4` in depth mode
- do not reload the source or reset the preview wipe

- [ ] **Step 6: Run the Tabbit acceptance check**

Expected: rail visible, depth panel switches correctly, view labels update, no overflow at 390, 1024, and 1440.

- [ ] **Step 7: Commit**

```powershell
git add cinedit/ui.html README.md
git commit -m "feat: add depth mode workspace"
```

---

### Task 7: Package, Benchmark, And Verify Offline

**Files:**
- Modify: `CineEdit.spec`
- Modify: `README.md`
- Create: `out/depth_benchmark.txt`

**Interfaces:**
- Consumes: model assets, Transformers, existing PyInstaller build.
- Produces: offline `dist/CineEdit` and measured performance data.

- [ ] **Step 1: Update PyInstaller data collection**

Add the depth model directory and Transformers package data:

```python
from PyInstaller.utils.hooks import collect_all

datas = [
    ("assets", "assets"),
    ("assets/weights/depth_anything_v2_small", "assets/weights/depth_anything_v2_small"),
    ("cinedit/ui.html", "cinedit"),
]

tmp_ret = collect_all("transformers")
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]
```

- [ ] **Step 2: Build and verify the packaged app**

Run:

```powershell
.venv\Scripts\python.exe -m PyInstaller CineEdit.spec --noconfirm --clean
```

Expected: build completes and `dist/CineEdit/CineEdit.exe` starts.

- [ ] **Step 3: Verify offline model loading**

Run the packaged app with network access disabled and confirm:

- `/api/env` reports the depth model available.
- First depth preview loads from the packaged asset directory.
- No Hugging Face download request is made.

- [ ] **Step 4: Benchmark**

Measure and write to `out/depth_benchmark.txt`:

```text
preview_540p_ms
render_720p_fps
render_1080p_fps
model_load_s
preset
device
```

If Fast is slower than the agreed speed-first target, lower the default input preset to 336 and repeat the benchmark.

- [ ] **Step 5: Run the full verification suite**

Run:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m compileall -q cinedit tests
node -e "const fs=require('fs');const s=fs.readFileSync('cinedit/ui.html','utf8');const m=s.match(/<script>([\s\S]*)<\/script>/);new Function(m[1]);console.log('JS_SYNTAX_OK')"
```

Expected: all tests pass, Python compiles, JavaScript syntax passes.

- [ ] **Step 6: Commit**

```powershell
git add CineEdit.spec README.md out/depth_benchmark.txt
git commit -m "build: package and benchmark depth mode"
```

## Plan Self-Review

- Spec coverage: architecture, model, depth processing, render pipeline, preview, UI, localization, error handling, packaging, tests, and acceptance criteria each map to a task.
- Placeholder scan: no TBD/TODO placeholders are used.
- Type consistency: `DepthConfig`, `DepthEngine.estimate`, `DepthRenderer.render`, `build_depth_cfg`, and `render_depth_preview` use the same names and argument order throughout.
