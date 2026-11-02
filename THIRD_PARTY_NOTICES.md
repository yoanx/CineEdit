# Third-Party Notices

CineEdit 的源代码使用 MIT License。项目依赖、模型权重和可选分发组件仍受各自
许可证约束。本文档用于说明主要第三方组件，不替代各项目发布的完整许可证文本。

## Runtime dependencies

| 组件 | 用途 | 许可证 |
| --- | --- | --- |
| PyTorch | GPU 张量计算与 CUDA 运行时 | BSD-3-Clause |
| torchvision | RAFT 光流模型与官方权重加载 | BSD-3-Clause |
| NumPy | 视频帧和数组处理 | BSD-3-Clause，并包含其他兼容许可证 |
| Pillow | 预览图像编码 | MIT-CMU |
| imageio-ffmpeg | FFmpeg 可执行文件查找与回退 | BSD-2-Clause |
| OpenCV | 视频深度模型输入缩放 | Apache-2.0 |
| Einops | 视频深度模型张量变换 | MIT |
| EasyDict | 视频深度模型配置对象 | MIT |
| pywebview | 桌面窗口与本地 Web UI | BSD-3-Clause |
| pythonnet | pywebview 的 .NET 互操作 | MIT |
| clr-loader | .NET 运行时加载 | MIT |

实际安装的版本和许可证以 `requirements.txt`、Python 包元数据及各上游项目为准。

## FFmpeg

源码仓库不提交 FFmpeg 可执行文件。项目生成的安装包可能附带一个静态 FFmpeg
构建。当前本地构建报告以下关键参数：

```text
ffmpeg version N-128100-gceb466cbcd-20261026
--enable-gpl --enable-version3 --enable-libx264 --enable-libx265
```

该构建受 GNU General Public License version 3 约束。发布包含该二进制的安装包时，
需要同时满足 FFmpeg 及其内置第三方库的许可证要求，包括提供相应许可证文本、
版权信息、构建配置和对应源码的获取方式。不要把这些义务与 CineEdit 自身的 MIT
许可证混为一谈。

参考资料：

- https://ffmpeg.org/legal.html
- https://ffmpeg.org/download.html
- https://www.gnu.org/licenses/gpl-3.0.html

## RAFT model weights

`raft_large` 和 `raft_small` 权重由 torchvision 的官方模型入口下载。源码仓库不提交
这些权重文件。打包分发时，请保留 PyTorch 和 torchvision 的版权与许可证声明，
并核对所使用模型权重的上游发布条款。

## Video Depth Anything Small

`Video-Depth-Anything-Small` 的推理代码和模型权重受 Apache License 2.0 约束。
项目内 vendored 代码位于 `cinedit/video_depth_anything/`，许可证文本随该目录
一并分发。模型权重位于 `assets/weights/video_depth_anything_vits/`。

## System components

- Microsoft Edge WebView2 Runtime 由 pywebview 的 Windows 后端使用。安装包通常
  依赖系统已有的 WebView2，不直接复制该运行时。
- CUDA、cuDNN、NVIDIA 驱动及相关动态库受 NVIDIA 的许可条款约束。

## Build and packaging tools

- PyInstaller 用于生成 Windows 可执行程序。
- Inno Setup 用于生成安装包。

这些工具通常不进入应用运行时，但再分发其产物时仍应遵守对应工具的许可证。
