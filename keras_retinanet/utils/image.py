"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import division
import keras
import numpy as np
import cv2
from PIL import Image
import imageio

from .transform import change_transform_origin


def read_image_bgr(path):
    """ Read an image in BGR format.

    Args
        path: Path to the image.
    """
    image = np.asarray(Image.open(path).convert('RGB'))
    return image[:, :, ::-1].copy()

def read_image_rgba(path):
    """ Read an image in RGBA format.

        Args
            path: Path to the image.
        """
    image = cv2.imread(path)
    return image

def read_image_fusion(path):
    """Read every channel of a fusion image.

    Args
        path: Path to the image.
    """
    image = imageio.imread(path)
    image[:, :, 0] = np.zeros((image.shape[0], image.shape[1]), dtype=int)
    #image[:, :, 0] = np.zeros((image.shape[0], image.shape[1]), dtype=int)
    return image[:, :, ::-1].copy()

def read_matrix_as_image(path):
    """Read every channel of a fusion npy matrix.

       Args
           path: Path to the image.
       """
    image = np.load(path)
    #img = np.zeros((image.shape[0], image.shape[1],6))
    #img[:,:,:5] = image
    #return img[:, :, ::-1].copy()
    return image[:, :, ::-1].copy()

def read_rgb_and_polar_images(path_rgb, path_polar):
    """Read an RGB image and its polarimetric equivalent.

           Args
               path_rgb: Path to the RGB image.
               path_polar: Path to the polarimetric image.
           """
    image_rgb = cv2.imread(path_rgb)
    image_rgb_rs = cv2.resize(image_rgb, dsize=(500, 500), interpolation=cv2.INTER_LANCZOS4)
    image_polar = cv2.imread(path_polar)
    image = np.zeros((500, 500, 6), dtype=int)
    image[:, :, :3] = image_rgb_rs
    image[:, :, 3:] = image_polar
    return image[:, :, ::-1].copy()

def read_rgb_and_polar_images_for_fusion(path_polar, path_rgb):
    """Read an RGB image and its polarimetric equivalent.

           Args
               path_rgb: Path to the RGB image.
               path_polar: Path to the polarimetric image.
           """
    """image_rgb = cv2.imread(path_rgb)
    image_rgb_rs = cv2.resize(image_rgb, dsize=(500, 500), interpolation=cv2.INTER_LANCZOS4)
    image_polar = cv2.imread(path_polar)
    if np.shape(image_polar)[2] == 3:
        image = np.zeros((500, 500, 7), dtype=int)
        image[:, :, :3] = image_rgb_rs
        image[:, :, 3:6] = image_polar
        image[:, :, 6] = image_polar[:, :, 2]
    elif np.shape(image_polar)[2] == 4:
        image = np.zeros((500, 500, 7), dtype=int)
        image[:, :, :3] = image_rgb_rs
        image[:, :, 3:] = image_polar
    return image[:, :, ::-1].copy()"""
    image_rgb = cv2.imread(path_rgb)
    image_polar = cv2.imread(path_polar, cv2.IMREAD_UNCHANGED)
    if image_rgb.shape[0] != image_polar.shape[0] and image_rgb.shape[1] != image_polar.shape[1]:
        image_rgb_rs = cv2.resize(image_rgb, dsize=(500, 500), interpolation=cv2.INTER_LANCZOS4)
    else:
        image_rgb_rs = image_rgb
    if np.shape(image_polar)[2] == 3:
        image = np.zeros((image_polar.shape[0], image_polar.shape[1], 7), dtype=int)
        image[:, :, :3] = image_rgb_rs
        image[:, :, 3:6] = image_polar
        image[:, :, 6] = image_polar[:, :, 2]
    elif np.shape(image_polar)[2] == 4:
        image = np.zeros((image_polar.shape[0], image_polar.shape[1], 7), dtype=int)
        image[:, :, :4] = image_polar
        image[:, :, 4:] = image_rgb_rs
        #image[:, :, 5] = np.zeros((image_polar.shape[0], image_polar.shape[1]), dtype=int)
    return image[:, :, ::-1].copy()

def preprocess_image(x, mode='caffe'):
    """ Preprocess an image by subtracting the ImageNet mean.

    Args
        x: np.array of shape (None, None, 3) or (3, None, None).
        mode: One of "caffe" or "tf".
            - caffe: will zero-center each color channel with
                respect to the ImageNet dataset, without scaling.
            - tf: will scale pixels between -1 and 1, sample-wise.

    Returns
        The input with the ImageNet mean subtracted.
    """
    # mostly identical to "https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py"
    # except for converting RGB -> BGR since we assume BGR already
    x = x.astype(keras.backend.floatx())
    if mode == 'tf':
        x /= 127.5
        x -= 1.
    elif mode == 'caffe':
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    return x

def preprocess_images(x, mode='caffe'):
    """ Preprocess an image by subtracting the ImageNet mean.

    Args
        x: np.array of shape (None, None, 3) or (3, None, None).
        mode: One of "caffe" or "tf".
            - caffe: will zero-center each color channel with
                respect to the ImageNet dataset, without scaling.
            - tf: will scale pixels between -1 and 1, sample-wise.

    Returns
        The input with the ImageNet mean subtracted.
    """
    # mostly identical to "https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py"
    # except for converting RGB -> BGR since we assume BGR already
    x[0] = x[0].astype(keras.backend.floatx())
    x[1] = x[1].astype(keras.backend.floatx())
    if mode == 'tf':
        x[0] /= 127.5
        x[0] -= 1.
        x[1] /= 127.5
        x[1] -= 1.
    elif mode == 'caffe':
        x[0][..., 0] -= 103.939
        x[0][..., 1] -= 116.779
        x[0][..., 2] -= 123.68
        x[1][..., 0] -= 103.939
        x[1][..., 1] -= 116.779
        x[1][..., 2] -= 123.68

    return x

def adjust_transform_for_image(transform, image, relative_translation):
    """ Adjust a transformation for a specific image.

    The translation of the matrix will be scaled with the size of the image.
    The linear part of the transformation will adjusted so that the origin of the transformation will be at the center of the image.
    """

    height, width, channels = image.shape

    result = transform

    # Scale the translation with the image size if specified.
    if relative_translation:
        result[0:2, 2] *= [width, height]

    # Move the origin of transformation.
    result = change_transform_origin(transform, (0.5 * width, 0.5 * height))

    return result


class TransformParameters:
    """ Struct holding parameters determining how to apply a transformation to an image.

    Args
        fill_mode:             One of: 'constant', 'nearest', 'reflect', 'wrap'
        interpolation:         One of: 'nearest', 'linear', 'cubic', 'area', 'lanczos4'
        cval:                  Fill value to use with fill_mode='constant'
        relative_translation:  If true (the default), interpret translation as a factor of the image size.
                               If false, interpret it as absolute pixels.
    """
    def __init__(
        self,
        fill_mode            = 'nearest',
        interpolation        = 'linear',
        cval                 = 0,
        relative_translation = True,
    ):
        self.fill_mode            = fill_mode
        self.cval                 = cval
        self.interpolation        = interpolation
        self.relative_translation = relative_translation

    def cvBorderMode(self):
        if self.fill_mode == 'constant':
            return cv2.BORDER_CONSTANT
        if self.fill_mode == 'nearest':
            return cv2.BORDER_REPLICATE
        if self.fill_mode == 'reflect':
            return cv2.BORDER_REFLECT_101
        if self.fill_mode == 'wrap':
            return cv2.BORDER_WRAP

    def cvInterpolation(self):
        if self.interpolation == 'nearest':
            return cv2.INTER_NEAREST
        if self.interpolation == 'linear':
            return cv2.INTER_LINEAR
        if self.interpolation == 'cubic':
            return cv2.INTER_CUBIC
        if self.interpolation == 'area':
            return cv2.INTER_AREA
        if self.interpolation == 'lanczos4':
            return cv2.INTER_LANCZOS4


def apply_transform(matrix, image, params):
    """
    Apply a transformation to an image.

    The origin of transformation is at the top left corner of the image.

    The matrix is interpreted such that a point (x, y) on the original image is moved to transform * (x, y) in the generated image.
    Mathematically speaking, that means that the matrix is a transformation from the transformed image space to the original image space.

    Args
      matrix: A homogeneous 3 by 3 matrix holding representing the transformation to apply.
      image:  The image to transform.
      params: The transform parameters (see TransformParameters)
    """
    output = cv2.warpAffine(
        image,
        matrix[:2, :],
        dsize       = (image.shape[1], image.shape[0]),
        flags       = params.cvInterpolation(),
        borderMode  = params.cvBorderMode(),
        borderValue = params.cval,
    )
    return output


def resize_image(img, min_side=800, max_side=1333):
    """ Resize an image such that the size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resized image.
    """
    (rows, cols, _) = img.shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale

def resize_images(img, min_side=800, max_side=1333):
    """ Resize an image such that the size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resized image.
    """
    (rows, cols, _) = img[0].shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    img[0] = cv2.resize(img[0], None, fx=scale, fy=scale)
    img[1] = cv2.resize(img[1], None, fx=scale, fy=scale)

    return img, scale

