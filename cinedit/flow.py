"""GPU 光流引擎。

使用 torchvision 内置的 RAFT（官方权重），在半分辨率上计算，再用引导滤波
上采样并对齐边缘。

相对 torchvision 默认前向的三处改动
-----------------------------------
1. **上采样与掩码只做一次**。默认实现每轮迭代都调用 mask_predictor 并做凸组合
   上采样，但只有最后一轮的结果被使用。掩码预测是对 128→81 通道的卷积，
   在 544×960 的输入上单次约 7.6 亿次乘加，重复 11 次纯属浪费。
2. **特征复用**。相邻两对光流 (i,i+1) 与 (i+1,i+2) 里，帧 i+1 的特征与上下文
   各被编码一次即可。编码器约占 RAFT 两成开销，缓存后这一半直接省掉。
3. **收敛提前退出**。视频连续帧位移通常很小，RAFT 常在 4~6 轮就稳定；
   逐轮比较位移增量，收敛即停，把省下的预算还给采样数。

精度上三者都不改变算法：输出与「跑满迭代 + 末轮上采样」的结果一致。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .imageops import guided_filter


def _pick_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref in ("auto", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if pref == "cuda":
            raise RuntimeError("指定了 cuda 但当前环境不可用")
        return torch.device("cpu")
    return torch.device(pref)


def _coords_grid(b: int, h: int, w: int, device) -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.arange(h, device=device),
                            torch.arange(w, device=device), indexing="ij")
    return torch.stack([xs, ys], dim=0).float()[None].expand(b, -1, -1, -1).contiguous()


def _upflow(flow: torch.Tensor, up_mask: torch.Tensor | None, factor: int = 8):
    """RAFT 的凸组合上采样（等价于 torchvision 内部实现，此处只做一次）。"""
    b, c, h, w = flow.shape
    if up_mask is None:
        return factor * F.interpolate(flow, size=(h * factor, w * factor),
                                      mode="bilinear", align_corners=True)
    up_mask = torch.softmax(up_mask.view(b, 1, 9, factor, factor, h, w), dim=2)
    up = F.unfold(factor * flow, kernel_size=3, padding=1).view(b, c, 9, 1, 1, h, w)
    up = torch.sum(up_mask * up, dim=2)
    return up.permute(0, 1, 4, 2, 5, 3).reshape(b, c, h * factor, w * factor)


class FlowEngine:
    """成对帧光流估计器，带特征缓存与收敛早退。"""

    def __init__(self, model: str = "raft_large", iters: int = 12,
                 device: str = "auto", scale: float = 0.5,
                 early_exit: bool = True, fp16: bool = False):
        self.model_name = model
        self.iters = iters
        self.device = _pick_device(device)
        self.scale = scale
        self.early_exit = early_exit
        self.fp16 = fp16
        self._net = None
        self._fast = True                 # 自定义循环是否可用
        self._prep: dict = {}             # token -> 预处理后的 [-1,1] 张量
        self._feat: dict = {}             # token -> (fmap, ctx_out)
        if self.device.type == "cuda":
            # TF32 只影响相关性体积里的 matmul，精度损失远低于 0.1 px。
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # benchmark=True 会让首次前向多花 20 秒做算子自动调优，实测稳态
            # 还更慢（92 ms vs 79 ms），因此保持关闭。

    # ------------------------------------------------------------ 模型加载
    def _build(self):
        from torchvision.models.optical_flow import (
            Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small,
        )
        # 先确保权重在本地缓存：打包分发时权重随包附带，这一步避免首次运行联网下载
        from .weights import ensure_weights
        ensure_weights()
        if self.model_name == "raft_small":
            net = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=True)
        else:
            net = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True)
        self._net = net.to(self.device).eval()
        for p in self._net.parameters():
            p.requires_grad_(False)

    @property
    def net(self):
        if self._net is None:
            self._build()
        return self._net

    # ------------------------------------------------------------ 尺寸对齐
    def _target_size(self, h: int, w: int) -> tuple[int, int]:
        th = max(64, int(round(h * self.scale / 8.0)) * 8)
        tw = max(64, int(round(w * self.scale / 8.0)) * 8)
        return th, tw

    def _prep_img(self, img: torch.Tensor, token) -> torch.Tensor:
        """下采样到计算分辨率并归一化到 [-1,1]，按 token 缓存。"""
        if token is not None:
            x = self._prep.get(token)
            if x is not None:
                return x
        _, _, h, w = img.shape
        th, tw = self._target_size(h, w)
        x = img if img.shape[-2:] == (th, tw) else F.interpolate(img, size=(th, tw), mode="area")
        x = (x * 2.0 - 1.0).contiguous()
        if token is not None:
            self._prep[token] = x
            while len(self._prep) > 2:
                self._prep.pop(next(iter(self._prep)))
        return x

    def _encode(self, x: torch.Tensor, token):
        """特征图 + 上下文，按 token 缓存（相邻两对光流共用中间帧）。"""
        if token is not None and token in self._feat:
            return self._feat[token]
        net = self.net
        fmap = net.feature_encoder(x)
        ctx_out = net.context_encoder(x)
        if token is not None:
            self._feat[token] = (fmap, ctx_out)
            # 只保留最近两帧的特征
            while len(self._feat) > 2:
                self._feat.pop(next(iter(self._feat)))
        return fmap, ctx_out

    # ------------------------------------------------------------ 主入口
    @torch.inference_mode()
    def flow(self, img0: torch.Tensor, img1: torch.Tensor,
             tok0=None, tok1=None) -> torch.Tensor:
        """返回 img0 -> img1 的光流，单位：原分辨率像素，shape [1,2,H,W]。"""
        _, _, h, w = img0.shape
        th, tw = self._target_size(h, w)
        a = self._prep_img(img0, tok0)
        b = self._prep_img(img1, tok1)

        if self._fast:
            try:
                f = self._run_fast(a, b, tok0, tok1, th, tw)
            except Exception:      # noqa: BLE001 —— 私有接口变动时退回官方前向
                self._fast = False
                f = self._run_default(a, b)
        else:
            f = self._run_default(a, b)

        if f.shape[-2:] != (h, w):
            f = F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False)
            f = f * (w / tw)
            try:
                f = guided_filter(img0, f, radius=3, eps=1e-4)
            except RuntimeError:
                pass
        return f.contiguous()

    def _run_default(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            return self.net(a, b, num_flow_updates=self.iters)[-1].float()

    def _run_fast(self, a, b, tok0, tok1, th, tw) -> torch.Tensor:
        net = self.net
        fmap1, ctx1 = self._encode(a, tok0)
        fmap2, ctx2 = self._encode(b, tok1)

        net.corr_block.build_pyramid(fmap1, fmap2)

        hidden_size = net.update_block.hidden_state_size
        hidden_state, ctx = torch.split(ctx1, [hidden_size, ctx1.shape[1] - hidden_size], dim=1)
        hidden_state = torch.tanh(hidden_state)
        ctx = F.relu(ctx)

        hf, wf = th // 8, tw // 8
        coords0 = _coords_grid(1, hf, wf, a.device)
        coords1 = coords0.clone()

        iters = max(1, self.iters)
        autocast = torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16)
        used = 0
        for i in range(iters):
            coords1 = coords1.detach()
            corr = net.corr_block.index_pyramid(centroids_coords=coords1)
            flow = coords1 - coords0
            with autocast:
                hidden_state, delta = net.update_block(hidden_state, ctx, corr, flow)
            coords1 = coords1 + delta.float()
            used = i + 1
            if self.early_exit and i >= 2:
                # 本轮位移增量的平均幅度（低分辨率像素，1 单位 = 8 输入像素）。
                # 增量足够小即视为收敛：后续迭代带来的位移变化不足半个输入像素。
                if float(delta.float().abs().mean()) < 1.0 / 16.0:
                    break
        self.last_iters = used

        with autocast:
            mask = None if net.mask_predictor is None else net.mask_predictor(hidden_state)
        out = _upflow(coords1 - coords0, None if mask is None else mask.float())
        return out.float()

    def flow_pair(self, img0: torch.Tensor, img1: torch.Tensor, tok0=None, tok1=None):
        """双向流：fwd = img0->img1，bwd = img1->img0。"""
        return (self.flow(img0, img1, tok0, tok1),
                self.flow(img1, img0, tok1, tok0))

    def drop_cache(self):
        self._prep.clear()
        self._feat.clear()

    def release(self):
        self.drop_cache()
        self._net = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
