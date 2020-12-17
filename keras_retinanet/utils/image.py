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
from scipy.signal import convolve2d
from skimage.filters.rank import entropy
from skimage.morphology import square

from .transform import change_transform_origin

def gaussian_pyramid(image, kernel, levels):
    """
    A function to create a Gaussian pyramid of a defined number of levels and from a chosen kernel.

    :param image: The image we want to use of dimension (N,M,3) or (M,N)
    :param kernel: The Gaussian kernel of dimention (k,k)
    :param levels: The desired number of levels in the Gaussian pyramid, an integer
    :return: The Gaussian pyramid, a list of numpy arrays
    """

    if len(np.shape(image)) == 3:
        gauss_l_r = image[:, :, 0]
        gauss_l_g = image[:, :, 1]
        gauss_l_b = image[:, :, 2]
    gauss_l = image
    pyramid = [gauss_l]
    for l in range(levels):
        if len(np.shape(image)) == 3:
            # channels last format
            gauss_l_r = downsample(gauss_l_r, kernel)
            gauss_l_g = downsample(gauss_l_g, kernel)
            gauss_l_b = downsample(gauss_l_b, kernel)
            gauss_l = np.zeros((gauss_l_b.shape[0], gauss_l_b.shape[1], 3))
            gauss_l[:, :, 0] = gauss_l_r
            gauss_l[:, :, 1] = gauss_l_g
            gauss_l[:, :, 2] = gauss_l_b
        else:
            gauss_l = downsample(gauss_l, kernel)
        pyramid.append(gauss_l)
    return pyramid

def laplacian_pyramid(image, kernel, levels):
    """
    A function to create a Laplacian pyramid of a defined number of levels and from a chosen kernel.

    :param image: The image we want to use of dimension (N,M,3) or (M,N)
    :param kernel: The Gaussian kernel of dimention (k,k)
    :param levels: The desired number of levels in the Laplacian pyramid, an integer
    :return: The Laplacian pyramid, a list of numpy arrays
    """

    gauss = gaussian_pyramid(image, kernel, levels)
    pyramid = []
    for l in range(len(gauss) - 2, -1, -1):
        if len(np.shape(image)) == 3:
            # channels last format
            gauss_l1r = upsample(gauss[l+1][:, :, 0])
            gauss_l1g = upsample(gauss[l+1][:, :, 1])
            gauss_l1b = upsample(gauss[l+1][:, :, 2])
            if gauss_l1r.shape[0] > gauss[l][:, :, 0].shape[0]:
                gauss_l1r = np.delete(gauss_l1r, -1, axis=0)
                gauss_l1g = np.delete(gauss_l1g, -1, axis=0)
                gauss_l1b = np.delete(gauss_l1b, -1, axis=0)
            if gauss_l1r.shape[1] > gauss[l][:, :, 0].shape[1]:
                gauss_l1r = np.delete(gauss_l1r, -1, axis=1)
                gauss_l1g = np.delete(gauss_l1g, -1, axis=1)
                gauss_l1b = np.delete(gauss_l1b, -1, axis=1)
            lap_l_r = gauss[l][:, :, 0] - gauss_l1r
            lap_l_g = gauss[l][:, :, 1] - gauss_l1g
            lap_l_b = gauss[l][:, :, 2] - gauss_l1b
            lap_l = np.zeros((lap_l_r.shape[0], lap_l_r.shape[1], 3))
            lap_l[:, :, 0] = lap_l_r
            lap_l[:, :, 1] = lap_l_g
            lap_l[:, :, 2] = lap_l_b
        else:
            gauss_l1 = upsample(gauss[l+1])
            if gauss_l1.shape[0] > gauss[l].shape[0]:
                gauss_l1 = np.delete(gauss_l1, -1, axis=0)
            if gauss_l1.shape[1] > gauss[l].shape[1]:
                gauss_l1 = np.delete(gauss_l1, -1, axis=1)
            lap_l = gauss[l] - gauss_l1
        pyramid.append(lap_l)
    return pyramid

def fused_laplacian_pyramid(gauss_pyramid_mod1, gauss_pyramid_mod2, lap_pyramid_mod1, lap_pyramid_mod2):
    """
    A funtion that builds a fused Laplacian pyramid of two modalities of the same image

    :param gauss_pyramid_mod1: The Gaussian pyramid of modality 1, a list of grayscale images, the first one in highest resolution
    :param gauss_pyramid_mod2: The Gaussian pyramid of modality 2, a list of grayscale images, the first one in highest resolution
    :param lap_pyramid_mod1: The Laplacian pyramid of modality 1, a list of grayscale images, the last one in highest resolution
    :param lap_pyramid_mod2: The Laplacian pyramid of modality 2, a list of grayscale images, the last one in highest resolution
    :return: The fused Laplacian pyramid of two modalities, a list of grayscale images, the last one in highest resolution,
    """

    fused_laplacian = []
    len_lap = len(lap_pyramid_mod1)
    for l in range(len_lap):
        fused_laplacian_temp = gauss_pyramid_mod1[len_lap-l-1]*lap_pyramid_mod1[l] + gauss_pyramid_mod2[len_lap-l-1]*lap_pyramid_mod2[l]
        fused_laplacian.append(fused_laplacian_temp)
    return fused_laplacian

def collapse_pyramid(lap_pyramid, gauss_pyramid):
    """
    A function to collapse a Laplacian pyramid in order to recover the enhanced image

    :param lap_pyramid: A Laplacian pyramid, a list of grayscale images, the last one in highest resolution
    :param gauss_pyramid: A Gaussian pyramid, a list of grayscale images, the last one in lowest resolution
    :return: A grayscale image
    """

    image = lap_pyramid[0]
    if len(np.shape(image)) == 3:
        im_r = upsample(gauss_pyramid[-1][:, :, 0])
        im_g = upsample(gauss_pyramid[-1][:, :, 1])
        im_b = upsample(gauss_pyramid[-1][:, :, 2])
        if im_r.shape[0] > image.shape[0]:
            im_r = np.delete(im_r, -1, axis=0)
            im_g = np.delete(im_g, -1, axis=0)
            im_b = np.delete(im_b, -1, axis=0)
        if im_r.shape[1] > image.shape[1]:
            im_r = np.delete(im_r, -1, axis=1)
            im_g = np.delete(im_g, -1, axis=1)
            im_b = np.delete(im_b, -1, axis=1)
        gauss = np.zeros((im_r.shape[0], im_r.shape[1], 3))
        gauss[:, :, 0] = im_r
        gauss[:, :, 1] = im_g
        gauss[:, :, 2] = im_b
    else:
        gauss = upsample(gauss_pyramid[-1])
        if gauss.shape[0] > image.shape[0]:
            gauss = np.delete(gauss, -1, axis=0)
        if gauss.shape[1] > image.shape[1]:
            gauss = np.delete(gauss, -1, axis=1)
    image = image + gauss
    for l in range(1,len(lap_pyramid),1):
        if len(np.shape(image)) == 3:
            im_r = upsample(image[:, :, 0])
            im_g = upsample(image[:, :, 1])
            im_b = upsample(image[:, :, 2])
            if im_r.shape[0] > lap_pyramid[l].shape[0]:
                im_r = np.delete(im_r, -1, axis=0)
                im_g = np.delete(im_g, -1, axis=0)
                im_b = np.delete(im_b, -1, axis=0)
            if im_r.shape[1] > lap_pyramid[l].shape[1]:
                im_r = np.delete(im_r, -1, axis=1)
                im_g = np.delete(im_g, -1, axis=1)
                im_b = np.delete(im_b, -1, axis=1)
            pyr_upsampled = np.zeros((im_r.shape[0], im_r.shape[1], 3))
            pyr_upsampled[:, :, 0] = im_r
            pyr_upsampled[:, :, 1] = im_g
            pyr_upsampled[:, :, 2] = im_b
        else:
            pyr_upsampled = upsample(image)
            if pyr_upsampled.shape[0] > lap_pyramid[l].shape[0]:
                pyr_upsampled = np.delete(pyr_upsampled, -1, axis=0)
            if pyr_upsampled.shape[1] > lap_pyramid[l].shape[1]:
                pyr_upsampled = np.delete(pyr_upsampled, -1, axis=1)
        image = lap_pyramid[l] + pyr_upsampled
    return image

def convolve(image, kernel):
    """
    A fonction to perform a 2D convolution operation over an image using a chosen kernel.

    :param image: The grayscale image we want to use of dimension (N,M)
    :param kernel: The convolution kernel of dimention (k,k)
    :return: The convolved image of dimension (N,M)
    """
    im_out = convolve2d(image, kernel, mode='same', boundary='symm')
    return im_out

def downsample(image, kernel):
    """
    A function to downsample an image.

    :param image: The grayscale image we want to use of dimension (N,M)
    :param kernel: The Gaussian blurring kernel of dimention (k,k)
    :return: The downsampled image of dimension (N/factor,M/factor)
    """
    blur_image = convolve(image, kernel)
    img_downsampled = blur_image[::2, ::2]
    return img_downsampled

def upsample(image):
    """

    :param image: The grayscale image we want to use of dimension (N,M)
    :param factor: The upsampling factor, an integer
    :return: The upsampled image of dimension (N*factor,M*factor)
    """

    #kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/12
    kernel = smooth_gaussian_kernel(0.4)

    img_upsampled = np.zeros((image.shape[0]*2, image.shape[1]*2), dtype=np.float64)
    img_upsampled[::2, ::2] = image[:, :]
    img_upsampled = 4 * convolve(img_upsampled, kernel)
    return img_upsampled

def classical_gaussian_kernel(k, sigma):
    """
    A function to generate a classical Gaussian kernel

    :param k: The size of the kernel, an integer
    :param sigma: variance of the gaussian distribution
    :return: A Gaussian kernel, a numpy array of shape (k,k)
    """
    w = np.linspace(-(k - 1) / 2, (k - 1) / 2, k)
    x, y = np.meshgrid(w, w)
    kernel = 0.5*np.exp(-0.5*(x**2 + y**2)/(sigma**2))/(np.pi*sigma**2)
    return kernel

def smooth_gaussian_kernel(a):
    """
     A 5*5 gaussian kernel to perform smooth filtering.

    :param a: the coefficient of the smooth filter. A float usually within [0.3, 0.6]
    :return: A smoothing Gaussian kernel, a numpy array of shape (5,5)
    """
    w = np.array([0.25 - a/2.0, 0.25, a, 0.25, 0.25 - a/2.0])
    kernel = np.outer(w, w)
    return kernel

def normalized_local_entropy(image, window_size):
    """
    A fonction that computes the local entropy given an image and a window size

    :param image: The grayscale image
    :param window_size: The size of the window that determines the neighbourhood of a pixel, an integer
    :return: The local entropy of the image, a grayscale image
    """

    local_entropy = entropy(image, square(window_size))
    return local_entropy

def local_contrast(image, window_size):
    """
     A fonction that computes the local contrast given an image and a window size

    :param image: The grayscale image
    :param window_size: The size of the window that determines the neighbourhood of a pixel, an integer
    :return: The local contrast of the image, a grayscale image
    """

    conv_filter = np.ones((window_size,window_size), dtype=int)
    local_mean = convolve(image, conv_filter)/(window_size**2)
    contrast = np.zeros((image.shape[0], image.shape[1]))
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            patch = image[max(0, x-int(window_size/2)):min(image.shape[0], x+int(window_size/2)), max(0, y-int(window_size/2)):min(image.shape[1], y+int(window_size/2))]
            patch = np.square(patch - local_mean[x,y])
            contrast[x,y] = np.sqrt(np.sum(patch)/(window_size**2))
    return contrast

def exposedness(image, sigma=0.2):
    """
    A fonction that computes the exposedness

    :param image:  The grayscale image
    :param sigma: A float, it is recommanded to set this value to 0.2
    :return: The exposedness of the image, a grayscale image.
    """

    exposedness = np.exp(-np.square(image - 0.5)/(2*sigma**2))
    return exposedness

def visibility(image, kernel1, kernel2):
    """
    A fonction that computes the visibility of an image given an image and two gaussian kernel

    :param image: The grayscale image
    :param kernel1: The gaussian kernel to compute the blurred image
    :param kernel2: The gaussian kernel to perform the final step of the visibility
    :return: The visibility, a grayscale image
    """

    img_blur = convolve(image, kernel1)
    visibility = np.sqrt(convolve(np.square(image - img_blur), kernel2))
    return visibility

def weight_combination(entropy, contrast, visibility, alpha1, alpha2, alpha3):
    """
    Combining the entropy, the contrast and the visibility to build a weight layer

    :param entropy: The local entropy of the image, a grayscale image
    :param contrast: The local contrast of the image, a grayscale image
    :param visibility: The visibility of the image, a grayscale image
    :param alpha1: The weight of the local entropy, a float within [0, 1]
    :param alpha2: The weight of the local contrast, a float within [0, 1]
    :param alpha3: The weight of the visibility, a float within [0, 1]
    :return: Weight map of the image, a grayscale image
    """

    weight = entropy**alpha1 * contrast**alpha2 * visibility**alpha3
    return weight

def weight_normalization(weight1, weight2):
    """
    A function to normalize the weights of each modality so the weights' sum is 1 for each pixel of the image

    :param weght1: The weight of madality 1, a grayscale image
    :param weight2: The weight of modality 2, a grayscale image
    :return: Two weights, weight1_normalized and weight2_normalized, respectively the normalized versions of weight1 and weight2, two grayscale images.
    """

    weight1_normalized = weight1 / (weight1 + weight2)
    weight2_normalized = weight2 / (weight1 + weight2)
    return weight1_normalized, weight2_normalized

def convert_image_to_floats(image):
    """
    A function to convert an image to a numpy array of floats within [0, 1]

    :param image: The image to be converted
    :return: The converted image
    """

    if np.max(image) <= 1.0:
        return image
    else:
        return image / 255.0

def pyramid_fusion(im_intensities, im_dop):
    im_intensities = cv2.imread(im_intensities)
    im_dop = cv2.imread(im_dop)
    kernel = smooth_gaussian_kernel(0.4)
    levels = 4
    window_size = 5

    im_mod1 = convert_image_to_floats(im_intensities[:, :, 2])
    im_mod2 = convert_image_to_floats(im_dop[:, :, 1])

    # kernels to compute visibility
    kernel1 = classical_gaussian_kernel(5, 2)
    kernel2 = classical_gaussian_kernel(5, 2)

    # Computation of local entropy, local contrast and visibility for value channel
    local_entropy_mod1 = normalized_local_entropy(im_mod1, window_size)
    #local_contrast_mod1 = local_contrast(im_mod1, window_size)
    visibility_mod1 = visibility(im_mod1, kernel1, kernel2)
    exposedness_mod1 = exposedness(im_mod1)
    # Combination of local entropy, local contrast and visibility for value channel
    weight_mod1 = weight_combination(local_entropy_mod1, exposedness_mod1, visibility_mod1, 1, 1, 1)

    # Computation of local entropy, local contrast and visibility for value channel
    local_entropy_mod2 = normalized_local_entropy(im_mod2, window_size)
    #local_contrast_mod2 = local_contrast(im_mod2, window_size)
    exposedness_mod2 = exposedness(im_mod2)
    visibility_mod2 = visibility(im_mod2, kernel1, kernel2)
    # Combination of local entropy, local contrast and visibility for value channel
    weight_mod2 = weight_combination(local_entropy_mod2, exposedness_mod2, visibility_mod2, 1, 1, 1)

    # Normalising weights of value channel and IR image
    weightN_mod1, weightN_mod2 = weight_normalization(weight_mod1, weight_mod2)

    # Creating Gaussian pyramids of the weights maps of respectively the value channel and IR image
    gauss_pyr_mod1_weights = gaussian_pyramid(weightN_mod1, kernel, levels)
    gauss_pyr_mod2_weights = gaussian_pyramid(weightN_mod2, kernel, levels)

    # Creating Laplacian pyramids of respectively the value channel and IR image
    lap_pyr_mod1 = laplacian_pyramid(im_mod1, kernel, levels)
    lap_pyr_mod2 = laplacian_pyramid(im_mod2, kernel, levels)

    # Creating the fused Laplacian of the two modalities
    lap_pyr_fusion = fused_laplacian_pyramid(gauss_pyr_mod1_weights, gauss_pyr_mod2_weights, lap_pyr_mod1, lap_pyr_mod2)

    # Creating the Gaussian pyramid of value channel in order to collapse the fused Laplacian pyramid
    gauss_pyr_mod1 = gaussian_pyramid(im_mod1, kernel, levels)
    collapsed_image = collapse_pyramid(lap_pyr_fusion, gauss_pyr_mod1)

    im_intensities[:, :, 2] = collapsed_image

    return im_intensities[:, :, ::-1].copy()

def read_image_entropy(path):
    image = cv2.imread(path)
    window_size = 5
    ent_ch1 = normalized_local_entropy(image[:, :, 0], window_size)
    ent_ch2 = normalized_local_entropy(image[:, :, 1], window_size)
    ent_ch3 = normalized_local_entropy(image[:, :, 2], window_size)

    entropy_image = image.copy()
    entropy_image[:, :, 0] = ent_ch1
    entropy_image[:, :, 1] = ent_ch2
    entropy_image[:, :, 2] = ent_ch3

    return entropy_image[:, :, ::-1].copy()

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
    #min_side = 800, max_side=1333
    """ Resize an image such that the size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resized image.
    """
    (rows1, cols1, _) = img[0].shape
    (rows2, cols2, _) = img[1].shape

    smallest_side1 = min(rows1, cols1)
    smallest_side2 = min(rows2, cols2)

    # rescale the image so the smallest side is min_side
    scale1 = min_side / smallest_side1
    scale2 = min_side / smallest_side2

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side1 = max(rows1, cols1)
    largest_side2 = max(rows2, cols2)
    if largest_side1 * scale1 > max_side:
        scale1 = max_side / largest_side1
    if largest_side2 * scale2 > max_side:
        scale2 = max_side / largest_side2

    # resize the image with the computed scale
    img[0] = cv2.resize(img[0], None, fx=scale1, fy=scale1)
    img[1] = cv2.resize(img[1], None, fx=scale2, fy=scale2)

    return img, [scale1, scale2]
