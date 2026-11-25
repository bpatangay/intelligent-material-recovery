import cv2
import numpy as np


class PhotometricPreprocessor:
    """
    Illumination normalization and specular glare suppression pipeline.
    Uses gentle L* CLAHE (clip_limit=1.2) to avoid amplifying conveyor belt noise,
    followed by edge-preserving bilateral filtering.
    """

    def __init__(
        self,
        clip_limit: float = 1.2,
        tile_grid_size: tuple[int, int] = (8, 8),
        bilateral_d: int = 5,
        sigma_color: float = 35.0,
        sigma_space: float = 35.0,
    ):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.bilateral_d = bilateral_d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def __call__(self, bgr_frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)

        l_norm = self.clahe.apply(l_chan)
        bgr_norm = cv2.cvtColor(cv2.merge((l_norm, a_chan, b_chan)), cv2.COLOR_LAB2BGR)

        return cv2.bilateralFilter(
            bgr_norm,
            d=self.bilateral_d,
            sigmaColor=self.sigma_color,
            sigmaSpace=self.sigma_space,
        )