"""散射（splat）通路：补齐 gather 采样拿不到的遮挡区域。

问题
----
gather 是"对每个输出像素问：你来自哪儿"。当物体扫过背景时，背景像素自身的
位移接近零，于是整个曝光区间都采到背景 —— 物体经过时的颜色彻底丢失，
形成一条规则的硬边（RSMB 的典型破绽）。

解法
----
补一条反向通路：让源帧里**运动中的像素**沿自身轨迹把颜色"撒"到输出上。
静止像素不参与（否则会污染背景），因此只在被扫过的带状区域产生贡献。

合成
----
令 c 为散射覆盖率（该像素在整个快门上被运动物体覆盖的时间比例），
G 为 gather 结果，O 为散射颜色：

    out = (1-c)·G + c·O

c=0 时退化为纯 gather；c=1 时是纯物体；中间值正好还原真实相机的
"半透明"拖影 —— 这正是复杂遮挡场景里 gather 独占方案无法表达的部分。
"""
from __future__ import annotations

import torch


def splat_accumulate(img: torch.Tensor, disp: torch.Tensor,
                     moving: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """把 img 按 disp 双线性散射到输出网格。

    img:    [1,3,H,W] 线性光
    disp:   [1,2,H,W] 每个源像素要移动到的目标位移（像素）
    moving: [1,1,H,W] 运动掩码（0..1），静止像素权重为 0

    返回 (acc[3,H*W], w[1,H*W])，由调用方归一化。
    """
    _, c, h, w = img.shape
    dev, dt = img.device, img.dtype

    xs = torch.arange(w, device=dev, dtype=dt)
    ys = torch.arange(h, device=dev, dtype=dt)
    gx = xs.view(1, 1, 1, w)
    gy = ys.view(1, 1, h, 1)

    x = (gx + disp[:, 0:1]).clamp(0, w - 1.001)
    y = (gy + disp[:, 1:2]).clamp(0, h - 1.001)

    x0 = torch.floor(x)
    y0 = torch.floor(y)
    fx = (x - x0).clamp(0, 1)
    fy = (y - y0).clamp(0, 1)
    x0 = x0.long()
    y0 = y0.long()
    x1 = (x0 + 1).clamp(max=w - 1)
    y1 = (y0 + 1).clamp(max=h - 1)

    base = (y0 * w + x0)
    acc = torch.zeros(3, h * w, device=dev, dtype=torch.float32)
    wt = torch.zeros(1, h * w, device=dev, dtype=torch.float32)

    img_f = img[0].reshape(c, h * w)
    w_f = moving.reshape(1, h * w)

    corners = (
        (base, (1 - fx) * (1 - fy)),
        (base + 1, fx * (1 - fy)),
        (base + w, (1 - fx) * fy),
        (base + w + 1, fx * fy),
    )
    for idx, kw in corners:
        k = (kw.reshape(1, h * w) * w_f)
        idx_f = idx.reshape(h * w)
        acc.index_add_(1, idx_f, img_f * k)
        wt.index_add_(1, idx_f, k)
    return acc, wt


@torch.inference_mode()
def render_disocclusion(prev_l: torch.Tensor, cur_l: torch.Tensor, nxt_l: torch.Tensor,
                        f_prev: torch.Tensor, f_next: torch.Tensor,
                        taus: list[float], strength: float = 1.0):
    """对给定时刻集合做散射积分，返回 (覆盖颜色 O[1,3,H,W], 覆盖率 c[1,1,H,W])。"""
    h, w = cur_l.shape[-2:]
    n = max(1, len(taus))

    acc = torch.zeros(3, h * w, device=cur_l.device, dtype=torch.float32)
    wt = torch.zeros(1, h * w, device=cur_l.device, dtype=torch.float32)

    for tau in taus:
        for img_l, flow, t_s in ((nxt_l, f_next, 1.0), (prev_l, f_prev, -1.0)):
            if (t_s > 0 and tau <= 0) or (t_s < 0 and tau > 0):
                continue
            disp = flow * ((tau - t_s) * strength)
            moving = (flow.pow(2).sum(dim=1, keepdim=True).sqrt() > 0.75).to(cur_l.dtype)
            a, b = splat_accumulate(img_l, disp, moving)
            acc = acc + a
            wt = wt + b

    out = acc / wt.clamp_min(1e-6)
    cov = (wt / float(n)).clamp(0.0, 1.0)
    return out.reshape(1, 3, h, w), cov.reshape(1, 1, h, w)
