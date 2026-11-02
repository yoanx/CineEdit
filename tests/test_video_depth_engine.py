import unittest

import numpy as np
import torch

from cinedit.video_depth_engine import VideoDepthEngine


class FakeVideoModel:
    def __init__(self):
        self.calls = []

    def infer_video_depth_one(self, frame, input_size, device, fp32):
        self.calls.append((frame.shape, input_size, device, fp32))
        return np.ones(frame.shape[:2], dtype=np.float32)


class VideoDepthEngineTests(unittest.TestCase):
    def test_estimate_converts_tensor_to_uint8_rgb(self):
        model = FakeVideoModel()
        engine = VideoDepthEngine(device="cpu", model=model)
        frame = torch.zeros(1, 3, 2, 3)

        depth = engine.estimate(frame, 392)

        self.assertEqual(tuple(depth.shape), (1, 1, 2, 3))
        self.assertEqual(model.calls[0][0], (2, 3, 3))
        self.assertEqual(model.calls[0][1], 392)
        self.assertEqual(model.calls[0][2], "cpu")
        self.assertTrue(model.calls[0][3])
        self.assertTrue(torch.equal(depth, torch.ones(1, 1, 2, 3)))


if __name__ == "__main__":
    unittest.main()
