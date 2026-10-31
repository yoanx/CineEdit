"""单帧运动模糊渲染：快门区间内的子帧采样积分。

物理模型
--------
真实相机在曝光时间 T 内把连续光强积分到传感器上：

    I_out(x) = (1/T) ∫_0^T I(x, t) dt

离散化后在快门区间 [-s/2, +s/2]（s 为快门角占帧间隔的比例）内取 N 个子帧。
关键几何关系（F 为 cur→next 的位移）：

    t = τ 时位于 x 的材质点   在 cur 中位于 x - τF
                              在 next 中位于 x + (1-τ)F
                              在 prev 中位于 x - (1+τ)F

注意不是 x + τF —— 那会把轨迹拉成两倍长（常见实现错误）。

性能结构
--------
1. 输入为**线性光**张量（由 pipeline 的帧缓存提供），本函数不做色彩转换。
   同一帧在一次渲染里会被当作 prev / cur / next 用三次，缓存后只转换一次。
2. 每个子帧的权重和恒等于 1（正半轴 (1-τ)+τ，负半轴 (1+τ)+(-τ)），
   因此默认路径**不需要权重和图像**，直接除以样本数。只有开启遮挡或光度
   抑制时才退回逐像素归一化。
3. 重采样走 Warper 的复用缓冲：逐子帧只剩两次 [1,H,W,2] 的元素级运算，
   没有网格重建，也没有逐样本张量分配。
"""
from __future__ import annotations

import torch

from .imageops import (Warper, flow_nhwc, jitter_field, linear_to_srgb,
                       srgb_to_linear, warp)

_SCRATCH: dict[tuple, dict] = {}


def _scratch(h: int, w: int, device, dtype, n: int) -> dict:
    """逐尺寸复用的一组缓冲区：累加器、权重、位移缩放。"""
    key = (h, w, str(device), str(dtype), n)
    s = _SCRATCH.get(key)
    if s is None:
        s = {
            "acc": torch.zeros(1, 3, h, w, device=device, dtype=dtype),
            "tau": torch.zeros(1, h, w, device=device, dtype=dtype),
            "sa": torch.zeros(1, h, w, 1, device=device, dtype=dtype),
            "sb": torch.zeros(1, h, w, 1, device=device, dtype=dtype),
            "wa": torch.zeros(1, 1, h, w, device=device, dtype=dtype),
            "wb": torch.zeros(1, 1, h, w, device=device, dtype=dtype),
            "wsum": torch.zeros(1, 1, h, w, device=device, dtype=dtype),
            "warper": Warper(h, w, device, dtype),
        }
        _SCRATCH[key] = s
    return s


def _fb_weight(f_ab: torch.Tensor, f_ba: torch.Tensor, sigma: float) -> torch.Tensor:
    """前向-后向光流一致性 -> 软可见性权重：e(x) = || f_ab(x) + f_ba(x + f_ab(x)) ||。"""
    back = warp(f_ba, f_ab, mode="bilinear", padding="border")
    err = (f_ab + back).pow(2).sum(dim=1, keepdim=True).sqrt()
    return torch.exp(-err.pow(2) / (2.0 * sigma * sigma)).clamp_(0.0, 1.0)


def _photo_weight(a: torch.Tensor, b: torch.Tensor, sigma: float) -> torch.Tensor:
    """光度一致性：同一时刻两个来源重建出的像素应当一致，不一致即判为遮挡或光流失效。"""
    diff = (a - b).abs().mean(dim=1, keepdim=True)
    return torch.exp(-diff.pow(2) / (2.0 * sigma * sigma)).clamp_(0.0, 1.0)


def to_linear(prev, cur, nxt, cfg):
    """sRGB 帧 -> 线性光；关闭线性积分时原样返回。供 pipeline 的帧缓存调用。"""
    if not cfg.linear_light:
        return prev, cur, nxt
    return srgb_to_linear(prev), srgb_to_linear(cur), srgb_to_linear(nxt)


@torch.inference_mode()
def render_frame(
    prev: torch.Tensor,
    cur: torch.Tensor,
    nxt: torch.Tensor,
    f_nf: torch.Tensor,          # cur -> next 位移（像素）
    f_pb: torch.Tensor | None,   # cur -> prev 位移；None 时用 -f_nf 近似
    cfg,
    f_nb: torch.Tensor | None = None,   # next -> cur，仅双向模式提供
    f_pf: torch.Tensor | None = None,   # prev -> cur，仅双向模式提供
    engine: str = "flow",
    linear: tuple | None = None,        # 预先算好的 (prev_l, cur_l, nxt_l)
) -> torch.Tensor:
    """渲染一帧。prev/cur/nxt 为 [1,3,H,W] sRGB，返回同 shape 的 sRGB。"""
    s = min(1.0, cfg.shutter_fraction() * max(0.0, cfg.strength))
    if s <= 1e-6 or cfg.samples < 2:
        return cur

    h, w = cur.shape[-2:]
    n = cfg.samples
    half = s * 0.5
    n_neg = max(1, n // 2)
    n_pos = n - n_neg

    # ---------- RSMB 式线性基线（仅 A/B 评测用）----------
    if engine == "linear":
        step = s / n
        out = torch.zeros_like(cur)
        for j in range(n):
            out.add_(warp(cur, f_nf * (-half + step * (j + 0.5))))
        return (out / n).clamp_(0, 1)

    # ---------- 线性光 ----------
    if linear is not None:
        prev_l, cur_l, nxt_l = linear
    elif cfg.linear_light:
        prev_l, cur_l, nxt_l = to_linear(prev, cur, nxt, cfg)
    else:
        prev_l, cur_l, nxt_l = prev, cur, nxt

    if f_pb is None:
        f_pb = -f_nf

    sc = _scratch(h, w, cur.device, cur_l.dtype, n)
    acc = sc["acc"].zero_()
    wsum = sc["wsum"]
    warper: Warper = sc["warper"]

    nf_next = flow_nhwc(f_nf)
    nf_prev = flow_nhwc(f_pb)
    nf_cur = nf_next

    use_occ = cfg.occlusion > 0 and (f_nb is not None or f_pf is not None)
    use_photo = cfg.photometric > 0
    weighted = use_occ or use_photo

    w_next = w_prev = None
    if use_occ:
        if f_nb is not None:
            w_next = (1.0 - cfg.occlusion) + cfg.occlusion * _fb_weight(
                f_nf, f_nb, cfg.cons_sigma)
        if f_pf is not None:
            w_prev = (1.0 - cfg.occlusion) + cfg.occlusion * _fb_weight(
                f_pb, f_pf, cfg.cons_sigma)
    if weighted:
        wsum.zero_()

    jit = jitter_field(n, h, w, cur.device, cur_l.dtype) if cfg.jitter else None
    mode = "bicubic" if getattr(cfg, "bicubic", False) else "bilinear"

    def emit(a_img, b_img, nf_a, nf_b, sa, sb, wa, wb, wz):
        """一个子帧：a 取自相邻帧，b 取自中心帧。"""
        a = warper(a_img, nf_a, sa, mode)
        b = warper(b_img, nf_b, sb, mode)
        if not weighted:
            acc.addcmul_(b, wb).addcmul_(a, wa)
            return
        w_a = wa
        if use_photo:
            w_a = w_a * (1.0 - cfg.photometric
                         + cfg.photometric * _photo_weight(a, b, cfg.photo_sigma))
        if wz is not None:
            w_a = w_a * wz if use_occ else w_a
        acc.addcmul_(b, wb).addcmul_(a, w_a)
        wsum.add_(wb).add_(w_a)

    # ---------- 正半轴 τ ∈ (0, s/2]，来源 = next ----------
    step = half / n_pos
    tau_b, sa_b, sb_b = sc["tau"], sc["sa"], sc["sb"]
    wa_b, wb_b = sc["wa"], sc["wb"]
    for j in range(n_pos):
        if jit is None:
            tau = step * (j + 0.5)
            emit(nxt_l, cur_l, nf_next, nf_cur, 1.0 - tau, -tau,
                 wa_b.fill_(tau), wb_b.fill_(1.0 - tau), w_next)
            continue
        # tau = step·(j + u)，u 为蓝噪声偏移
        torch.mul(jit[j].view(1, h, w), step, out=tau_b).add_(step * j)
        t4 = tau_b.view(1, h, w, 1)
        torch.sub(1.0, t4, out=sa_b)          # a 的缩放 1-τ
        torch.neg(t4, out=sb_b)               # b 的缩放 -τ
        emit(nxt_l, cur_l, nf_next, nf_cur, sa_b, sb_b,
             tau_b.view(1, 1, h, w), torch.sub(1.0, tau_b.view(1, 1, h, w), out=wb_b),
             w_next)

    # ---------- 负半轴 τ ∈ [-s/2, 0)，来源 = prev ----------
    step = half / n_neg
    for j in range(n_neg):
        if jit is None:
            tau = -step * (n_neg - j - 0.5)
            emit(prev_l, cur_l, nf_prev, nf_cur, 1.0 + tau, -tau,
                 wa_b.fill_(-tau), wb_b.fill_(1.0 + tau), w_prev)
            continue
        torch.mul(jit[n_pos + j].view(1, h, w), step, out=tau_b).sub_(step * (n_neg - j))
        t4 = tau_b.view(1, h, w, 1)
        torch.add(1.0, t4, out=sa_b)          # a 的缩放 1+τ
        torch.neg(t4, out=sb_b)               # b 的缩放 -τ
        emit(prev_l, cur_l, nf_prev, nf_cur, sa_b, sb_b,
             torch.neg(tau_b, out=wa_b.view(1, h, w)).view(1, 1, h, w),
             torch.add(1.0, tau_b.view(1, 1, h, w), out=wb_b),
             w_prev)

    # ---------- 归一化 ----------
    if weighted:
        eps = 1e-4
        acc.add_(cur_l, alpha=eps)
        wsum.add_(eps)
        out_l = acc / wsum
    else:
        out_l = acc.mul_(1.0 / n)

    # ---------- 散射通路：回填被运动物体扫过的背景区域（实验性）----------
    if cfg.disocclusion > 0 and cfg.splat_samples > 0:
        from .splat import render_disocclusion
        n_s = min(cfg.splat_samples, n)
        taus = [-half + (2 * half) * (k + 0.5) / n_s for k in range(n_s)]
        O, c = render_disocclusion(prev_l, cur_l, nxt_l, -f_pb, f_nf, taus)
        diff = (O - out_l).abs().mean(dim=1, keepdim=True)
        gate = torch.clamp((diff - 0.02) / 0.06, 0.0, 1.0)
        c = (c * gate * cfg.disocclusion).clamp_(0.0, 1.0)
        out_l = (1.0 - c) * out_l + c * O

    return linear_to_srgb(out_l) if cfg.linear_light else out_l.clamp_(0, 1)
