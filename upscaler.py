import cv2
import gymnasium as gym


class UpscaleRender(gym.Wrapper):
    def __init__(self, env, scale=4):
        super().__init__(env)
        self.scale = scale

    def render(self):
        frame = self.env.render()
        h, w = frame.shape[:2]
        # Upscale using nearest-neighbor (pixel crisp)
        frame = cv2.resize(
            frame, (w * self.scale, h * self.scale), interpolation=cv2.INTER_NEAREST
        )
        return frame
