import numpy as np
from PIL import Image
from torchvision import transforms

from .registry import TRANSFORMS, BaseTransform


@TRANSFORMS.register_module()
class CenterCropArr(BaseTransform):
    """ADM-style resize followed by a center crop."""

    @staticmethod
    def apply(image: Image.Image, size: int) -> Image.Image:
        while min(*image.size) >= 2 * size:
            image = image.resize(tuple(length // 2 for length in image.size), resample=Image.BOX)
        scale = size / min(*image.size)
        image = image.resize(tuple(round(length * scale) for length in image.size), resample=Image.BICUBIC)
        array = np.asarray(image)
        crop_y = (array.shape[0] - size) // 2
        crop_x = (array.shape[1] - size) // 2
        return Image.fromarray(array[crop_y:crop_y + size, crop_x:crop_x + size])

    def __init__(self, keys, size):
        self.keys = list(keys)
        self.size = int(size)

    def __call__(self, results: dict) -> dict:
        for key in self.keys:
            image = results.get(key)
            if isinstance(image, Image.Image):
                results[key] = self.apply(image, self.size)
        return results


@TRANSFORMS.register_module()
class CenterCrop(BaseTransform):
    def __init__(self, keys, size=None):
        self.keys = list(keys)
        self.size = int(size) if size else None

    def __call__(self, results: dict) -> dict:
        for key in self.keys:
            image = results.get(key)
            if not isinstance(image, Image.Image):
                continue
            width, height = image.size
            crop_size = self.size or min(width, height)
            left = (width - crop_size) // 2
            top = (height - crop_size) // 2
            results[key] = image.crop((left, top, left + crop_size, top + crop_size))
        return results


@TRANSFORMS.register_module()
class RandomFlip(BaseTransform):
    def __init__(self, keys, prob=0.5, direction="horizontal"):
        self.keys = list(keys)
        self.prob = float(prob)
        self.direction = direction

    def __call__(self, results: dict) -> dict:
        import torch

        if torch.rand(1).item() >= self.prob:
            return results
        transpose = Image.FLIP_TOP_BOTTOM if self.direction == "vertical" else Image.FLIP_LEFT_RIGHT
        for key in self.keys:
            image = results.get(key)
            if isinstance(image, Image.Image):
                results[key] = image.transpose(transpose)
        return results


@TRANSFORMS.register_module()
class ToTensor(BaseTransform):
    def __init__(self, keys):
        self.keys = list(keys)
        self._to_tensor = transforms.ToTensor()

    def __call__(self, results: dict) -> dict:
        for key in self.keys:
            if key in results:
                results[key] = self._to_tensor(results[key])
        return results
