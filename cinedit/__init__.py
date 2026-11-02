"""CineEdit —— 基于光流子帧采样的物理级运动模糊。

与 ReelSmart Motion Blur 的核心差异：
1. 线性光空间积分（RSMB 在 gamma 空间平均，亮背景前会产生过量光晕）
2. 前向-后向光流一致性加权（抑制遮挡边界的背景拖影）
3. N 次子帧采样 + 逐像素抖动（消除长拖尾的阶梯状断层）
4. GPU 光流 + 引导滤波上采样（半分辨率算流，边缘对齐上采样）
"""

__version__ = "1.0.0"

from .config import BlurConfig
from .pipeline import MotionBlurRenderer

__all__ = ["BlurConfig", "MotionBlurRenderer", "__version__"]
