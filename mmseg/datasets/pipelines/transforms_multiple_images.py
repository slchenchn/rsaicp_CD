'''
Author: Shuailin Chen
Created Date: 2021-06-21
Last Modified: 2021-08-23
	content: 
'''
import mmcv
import numpy as np
from mmcv.utils import deprecated_api_warning, is_tuple_of
from numpy import random
from mmseg.utils import split_images
from PIL import ImageFilter
from PIL import Image

from ..builder import PIPELINES
from .transforms import PhotoMetricDistortion, Normalize

@PIPELINES.register_module()
class PhotoMetricDistortionMultiImages(PhotoMetricDistortion):
    ''' Apply photometric distortion to multiple images sequentially, see class PhotoMetricDistortion for detail
    '''
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def __call__(self, results):
        img = results['img']
        img1, img2 = split_images(img)

        result = dict(img=img1)
        img1 = super().__call__(result)['img']
        
        result = dict(img=img2)
        img2 = super().__call__(result)['img']
    
        results['img'] = np.concatenate((img1, img2), axis=-1)

        return results


# @PIPELINES.register_module()
# class PhotoMetricDistortionMultiImages(object):
#     """Apply photometric distortion to image sequentially, every transformation
#     is applied with a probability of 0.5. The position of random contrast is in
#     second or second to last.

#     1. random brightness
#     2. random contrast (mode 0)
#     3. convert color from BGR to HSV
#     4. random saturation
#     5. random hue
#     6. convert color from HSV to BGR
#     7. random contrast (mode 1)

#     Args:
#         brightness_delta (int): delta of brightness.
#         contrast_range (tuple): range of contrast.
#         saturation_range (tuple): range of saturation.
#         hue_delta (int): delta of hue.
#     """

#     def __init__(self,
#                  brightness_delta=32,
#                  contrast_range=(0.5, 1.5),
#                  saturation_range=(0.5, 1.5),
#                  hue_delta=18):
#         self.brightness_delta = brightness_delta
#         self.contrast_lower, self.contrast_upper = contrast_range
#         self.saturation_lower, self.saturation_upper = saturation_range
#         self.hue_delta = hue_delta

#     def convert(self, img, alpha=1, beta=0):
#         """Multiple with alpha and add beat with clip."""
#         img = img.astype(np.float32) * alpha + beta
#         img = np.clip(img, 0, 255)
#         return img.astype(np.uint8)

#     def brightness(self, img):
#         """Brightness distortion."""
#         if random.randint(2):
#             return self.convert(
#                 img,
#                 beta=random.uniform(-self.brightness_delta,
#                                     self.brightness_delta))
#         return img

#     def contrast(self, img):
#         """Contrast distortion."""
#         if random.randint(2):
#             return self.convert(
#                 img,
#                 alpha=random.uniform(self.contrast_lower, self.contrast_upper))
#         return img

#     def saturation(self, img):
#         """Saturation distortion."""
#         if random.randint(2):
#             img = mmcv.bgr2hsv(img)
#             img[:, :, 1] = self.convert(
#                 img[:, :, 1],
#                 alpha=random.uniform(self.saturation_lower,
#                                      self.saturation_upper))
#             img = mmcv.hsv2bgr(img)
#         return img

#     def hue(self, img):
#         """Hue distortion."""
#         if random.randint(2):
#             img = mmcv.bgr2hsv(img)
#             img[:, :,
#                 0] = (img[:, :, 0].astype(int) +
#                       random.randint(-self.hue_delta, self.hue_delta)) % 180
#             img = mmcv.hsv2bgr(img)
#         return img

#     def __call__(self, results):
#         img = results['img']
#         img1, img2 = split_images(img)

#         result = dict(img=img1)
#         img1 = self.vanilla_call(result)['img']
        
#         result = dict(img=img2)
#         img2 = self.vanilla_call(result)['img']
    
#         results['img'] = np.concatenate((img1, img2), axis=-1)
#         return results

#     def vanilla_call(self, results):
#         """Call function to perform photometric distortion on images.

#         Args:
#             results (dict): Result dict from loading pipeline.

#         Returns:
#             dict: Result dict with images distorted.
#         """

#         img = results['img']
#         # random brightness
#         img = self.brightness(img)

#         # mode == 0 --> do random contrast first
#         # mode == 1 --> do random contrast last
#         mode = random.randint(2)
#         if mode == 1:
#             img = self.contrast(img)

#         # random saturation
#         img = self.saturation(img)

#         # random hue
#         img = self.hue(img)

#         # random contrast
#         if mode == 0:
#             img = self.contrast(img)

#         results['img'] = img
#         return results

#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += (f'(brightness_delta={self.brightness_delta}, '
#                      f'contrast_range=({self.contrast_lower}, '
#                      f'{self.contrast_upper}), '
#                      f'saturation_range=({self.saturation_lower}, '
#                      f'{self.saturation_upper}), '
#                      f'hue_delta={self.hue_delta})')
#         return repr_str


@PIPELINES.register_module()
class NormalizeMultiImages(Normalize):
    """Normalize multiple images, see class Normalize for detail
    """

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def __call__(self, results):
        img = results['img']
        img1, img2 = split_images(img)

        result = dict(img=img1)
        result = super().__call__(result)
        img1 = result['img']
        img_norm_cfg = result['img_norm_cfg']
        
        result = dict(img=img2)
        img2 = super().__call__(result)['img']
    
        results['img'] = np.concatenate((img1, img2), axis=-1)
        results['img_norm_cfg'] = img_norm_cfg

        return results

    
@PIPELINES.register_module()
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

    def __init__(self, sigma_min=0, sigma_max=1):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, results):
        img = results['img']
        img1, img2 = split_images(img)

        img_result = []
        for img in (img1, img2):
            img = Image.fromarray(img)
            sigma = np.random.uniform(self.sigma_min, self.sigma_max)
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
            img_result.append(img)

        results['img'] = np.concatenate(img_result, axis=-1)
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str