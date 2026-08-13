from __future__ import annotations

import random

import torch


class RandAugment:
    def __init__(self, n: int = 2, m: int = 9):
        self._n = n
        self._m = m

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        for _ in range(self._n):
            op = random.choice([
                self._adjust_brightness,
                self._adjust_contrast,
                self._adjust_saturation,
                self._adjust_hue,
                self._posterize,
                self._solarize,
                self._equalize,
                self._rotate,
                self._translate_x,
                self._translate_y,
                self._shear_x,
                self._shear_y,
            ])
            img = op(img).clamp_(0.0, 1.0)
        return img

    def _scale(self) -> float:
        return self._m / 9.0

    def _adjust_brightness(self, img: torch.Tensor) -> torch.Tensor:
        factor = 1.0 + self._scale() * random.uniform(-0.5, 0.5)
        return (img * factor).clamp(0, 1)

    def _adjust_contrast(self, img: torch.Tensor) -> torch.Tensor:
        factor = 1.0 + self._scale() * random.uniform(-0.5, 0.5)
        mean = img.mean(dim=(1, 2), keepdim=True)
        return ((img - mean) * factor + mean).clamp(0, 1)

    def _adjust_saturation(self, img: torch.Tensor) -> torch.Tensor:
        factor = 1.0 + self._scale() * random.uniform(-0.5, 0.5)
        gray = img.mean(dim=0, keepdim=True)
        return (img * factor + gray * (1 - factor)).clamp(0, 1)

    def _adjust_hue(self, img: torch.Tensor) -> torch.Tensor:
        import torchvision.transforms.functional as TF
        factor = self._scale() * random.uniform(-0.2, 0.2)
        return TF.adjust_hue(img, factor)

    def _posterize(self, img: torch.Tensor) -> torch.Tensor:
        bits = int(8 - self._scale() * 7)
        shift = 8 - bits
        img_int = (img * 255).byte()
        img_int = (img_int >> shift) << shift
        return img_int.float() / 255.0

    def _solarize(self, img: torch.Tensor) -> torch.Tensor:
        threshold = 0.5 + self._scale() * 0.3
        mask = img < threshold
        result = img.clone()
        result[~mask] = 1.0 - result[~mask]
        return result

    def _equalize(self, img: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(img)
        for c in range(img.shape[0]):
            ch = img[c]
            hist = torch.histc(ch, bins=256, min=0, max=1)
            cdf = hist.cumsum(dim=0)
            cdf_norm = cdf / cdf[-1] * 255
            idx = (ch * 255).long().clamp(0, 255)
            result[c] = cdf_norm[idx] / 255.0
        return result

    def _rotate(self, img: torch.Tensor) -> torch.Tensor:
        import torchvision.transforms.functional as TF
        angle = self._scale() * random.uniform(-30, 30)
        return TF.rotate(img.unsqueeze(0), angle, fill=0.5).squeeze(0)

    def _translate_x(self, img: torch.Tensor) -> torch.Tensor:
        import torchvision.transforms.functional as TF
        dx = int(img.shape[2] * self._scale() * random.uniform(-0.2, 0.2))
        return TF.affine(img.unsqueeze(0), angle=0, translate=(dx, 0), scale=1.0, shear=0).squeeze(0)

    def _translate_y(self, img: torch.Tensor) -> torch.Tensor:
        import torchvision.transforms.functional as TF
        dy = int(img.shape[1] * self._scale() * random.uniform(-0.2, 0.2))
        return TF.affine(img.unsqueeze(0), angle=0, translate=(0, dy), scale=1.0, shear=0).squeeze(0)

    def _shear_x(self, img: torch.Tensor) -> torch.Tensor:
        import torchvision.transforms.functional as TF
        shear = self._scale() * random.uniform(-20, 20)
        return TF.affine(img.unsqueeze(0), angle=0, translate=(0, 0), scale=1.0, shear=shear).squeeze(0)

    def _shear_y(self, img: torch.Tensor) -> torch.Tensor:
        import torchvision.transforms.functional as TF
        shear = self._scale() * random.uniform(-20, 20)
        return TF.affine(img.unsqueeze(0), angle=0, translate=(0, 0), scale=1.0, shear=(0, shear)).squeeze(0)
