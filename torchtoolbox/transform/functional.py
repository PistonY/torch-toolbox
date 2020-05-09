# -*- coding: utf-8 -*-
from __future__ import division
from string import ascii_letters
import torch
import math
import cv2
import numpy as np
import random
import numbers
import collections

Sequence = collections.abc.Sequence
Iterable = collections.abc.Iterable

INTER_MODE = {
    'NEAREST': cv2.INTER_NEAREST,
    'BILINEAR': cv2.INTER_LINEAR,
    'BICUBIC': cv2.INTER_CUBIC}
PAD_MOD = {'constant': cv2.BORDER_CONSTANT,
           'edge': cv2.BORDER_REPLICATE,
           'reflect': cv2.BORDER_DEFAULT,
           'symmetric': cv2.BORDER_REFLECT
           }


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy(img):
    return isinstance(img, np.ndarray)


def _is_numpy_image(img):
    return img.ndim in {2, 3}


def to_tensor(pic):
    """Convert a ``numpy.ndarray`` image to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if _is_numpy_image(pic):
        if pic.ndim == 2:
            pic = cv2.cvtColor(pic, cv2.COLOR_GRAY2RGB)
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img
    else:
        raise TypeError('pic should be ndarray. Got {}.'.format(type(pic)))


def to_cv_image(pic, mode=None):
    """Convert a tensor or an ndarray to CV Image.


    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to CV Image.
        mode (str): color space and pixel depth of input data (optional).

    Returns:
        cv2 Image: Image converted to cv2 Image.
    """
    if not (_is_numpy_image(pic) or _is_tensor_image(pic)):
        raise TypeError(
            'pic should be Tensor or ndarray. Got {}.'.format(
                type(pic)))

    if isinstance(pic, torch.FloatTensor):
        pic = pic.mul(255).byte()
    if isinstance(pic, torch.Tensor):
        pic = pic.numpy().transpose((1, 2, 0)).squeeze()
    if not isinstance(pic, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                        'not {}'.format(type(pic)))
    if mode is not None:
        pic = cv2.cvtColor(pic, mode)
    return pic


def normalize(tensor, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
    return tensor


def resize(img, size, interpolation='BILINEAR'):
    r"""Resize the input CV Image to the given size.

    Args:
        img (CV Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``BILINEAR``

    Returns:
        PIL Image: Resized image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))
    if not (
        isinstance(
            size,
            int) or (
            isinstance(
                size,
                Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    interpolation = INTER_MODE[interpolation]
    if isinstance(size, int):
        w, h, _ = img.shape
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
    else:
        oh, ow = map(int, size)
    return cv2.resize(img, (ow, oh), interpolation=interpolation)


def pad(img, padding, fill=0, padding_mode='constant'):
    r"""Pad the given CV Image on all sides with specified padding mode and fill value.

    Args:
        img (CV Image): Image to be padded.
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

            - constant: pads with a constant value, this value is specified with fill

            - edge: pads with the last value on the edge of the image

            - reflect: pads with reflection of image (without repeating the last value on the edge)

                       padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                       will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image (repeating the last value on the edge)

                         padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                         will result in [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        CV Image: Padded image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')

    if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
        raise ValueError(
            "Padding must be an int or a 2, or 4 element tuple, not a " +
            "{} element tuple".format(
                len(padding)))

    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], \
        'Padding mode should be either constant, edge, reflect or symmetric'

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    if isinstance(padding, Sequence) and len(padding) == 4:
        pad_left, pad_top, pad_right, pad_bottom = padding

    if isinstance(fill, numbers.Number):
        fill = (fill,) * (2 * len(img.shape) - 3)

    if padding_mode == 'constant':
        assert (len(fill) == 3 and len(img.shape) == 3) or (len(fill) == 1 and len(img.shape) ==
                                                            2), 'channel of image is {} but length of fill is {}'.format(img.shape[-1], len(fill))
    img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                             PAD_MOD[padding_mode], value=fill)
    return img


def crop(img, i, j, h, w):
    """Crop the given CV Image.

    Args:
        img (PIL Image): Image to be cropped.
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped image.
        w (int): Width of the cropped image.

    Returns:
        PIL Image: Cropped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    return img[j:j + w, i:i + h, ...].copy()


def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    w, h, _ = img.shape
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)


def resized_crop(img, i, j, h, w, size, interpolation='BILINEAR'):
    """Crop the given CV Image and resize it to desired size.

    Args:
        img (CV Image): Image to be cropped.
        i (int): i in (i,j) i.e coordinates of the upper left corner
        j (int): j in (i,j) i.e coordinates of the upper left corner
        h (int): Height of the cropped image.
        w (int): Width of the cropped image.
        size (sequence or int): Desired output size. Same semantics as ``resize``.
        interpolation (int, optional): Desired interpolation. Default is
            ``BILINEAR``.
    Returns:
        CV Image: Cropped image.
    """
    assert _is_numpy_image(img), 'img should be CV Image'
    img = crop(img, i, j, h, w)
    img = resize(img, size, interpolation)
    return img


def hflip(img):
    """Horizontally flip the given CV Image.

    Args:
        img (CV Image): Image to be flipped.

    Returns:
        CV Image:  Horizontall flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    return cv2.flip(img, 0)


def vflip(img):
    """Vertically flip the given CV Image.

    Args:
        img (CV Image): Image to be flipped.

    Returns:
        CV Image:  Vertically flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    return cv2.flip(img, 1)


def _get_perspective_coeffs(
        h,
        w,
        shear,
        anglex,
        angley,
        anglez,
        scale,
        translate,
        fov):
    """
    This function is partly referred to https://blog.csdn.net/dcrmg/article/details/80273818
    """
    centery = h * 0.5
    centerx = w * 0.5

    alpha = math.radians(shear)
    beta = math.radians(anglez)

    lambda1 = scale[0]
    lambda2 = scale[1]

    tx = translate[0]
    ty = translate[1]

    sina = math.sin(alpha)
    cosa = math.cos(alpha)
    sinb = math.sin(beta)
    cosb = math.cos(beta)

    M00 = cosb * (lambda1 * cosa ** 2 + lambda2 * sina ** 2) - \
        sinb * (lambda2 - lambda1) * sina * cosa
    M01 = - sinb * (lambda1 * sina ** 2 + lambda2 * cosa ** 2) + \
        cosb * (lambda2 - lambda1) * sina * cosa

    M10 = sinb * (lambda1 * cosa ** 2 + lambda2 * sina ** 2) + \
        cosb * (lambda2 - lambda1) * sina * cosa
    M11 = + cosb * (lambda1 * sina ** 2 + lambda2 * cosa ** 2) + \
        sinb * (lambda2 - lambda1) * sina * cosa
    M02 = centerx - M00 * centerx - M01 * centery + tx
    M12 = centery - M10 * centerx - M11 * centery + ty
    affine_matrix = np.array(
        [[M00, M01, M02], [M10, M11, M12], [0, 0, 1]], dtype=np.float32)
    # -------------------------------------------------------------------------------
    z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(math.radians(fov / 2))

    radx = math.radians(anglex)
    rady = math.radians(angley)

    sinx = math.sin(radx)
    cosx = math.cos(radx)
    siny = math.sin(rady)
    cosy = math.cos(rady)

    r = np.array([[cosy, 0, -siny, 0],
                  [-siny * sinx, cosx, -sinx * cosy, 0],
                  [cosx * siny, sinx, cosx * cosy, 0],
                  [0, 0, 0, 1]])

    pcenter = np.array([centerx, centery, 0, 0], np.float32)

    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, h, 0, 0], np.float32) - pcenter
    p4 = np.array([w, h, 0, 0], np.float32) - pcenter

    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)

    list_dst = [dst1, dst2, dst3, dst4]

    org = np.array([[0, 0],
                    [w, 0],
                    [0, h],
                    [w, h]], np.float32)

    dst = np.zeros((4, 2), np.float32)

    for i in range(4):
        dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
        dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]

    perspective_matrix = cv2.getPerspectiveTransform(org, dst)
    matrix = perspective_matrix @ affine_matrix
    return matrix


def perspective(
    img, fov=45, anglex=0, angley=0, anglez=0, shear=0, translate=(
        0, 0), scale=(
            1, 1), resample='BILINEAR', fillcolor=(
                0, 0, 0)):
    """Perform perspective transform of the given CV Image.
    This function is partly referred to https://blog.csdn.net/dcrmg/article/details/80273818

    Returns:
        PIL Image:  Perspectively transformed Image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))
    imgtype = img.dtype
    gray_scale = False

    if len(img.shape) == 2:
        gray_scale = True
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    h, w, _ = img.shape
    matrix = _get_perspective_coeffs(
        h,
        w,
        shear,
        anglex,
        angley,
        anglez,
        scale,
        translate,
        fov)
    img = cv2.warpPerspective(
        img,
        matrix,
        (w,
         h),
        flags=INTER_MODE[resample],
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fillcolor)
    if gray_scale:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.astype(imgtype)


def five_crop(img, size):
    """Crop the given CV Image into four corners and the central crop.

    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
       size (sequence or int): Desired output size of the crop. If size is an
           int instead of sequence like (h, w), a square crop (size, size) is
           made.

    Returns:
       tuple: tuple (tl, tr, bl, br, center)
                Corresponding top left, top right, bottom left, bottom right and center crop.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(
            size) == 2, "Please provide only two dimensions (h, w) for size."

    w, h, _ = img.shape
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError(
            "Requested crop size {} is bigger than input size {}".format(
                size, (h, w)))
    tl = img.crop((0, 0, crop_w, crop_h))
    tr = img.crop((w - crop_w, 0, w, crop_h))
    bl = img.crop((0, h - crop_h, crop_w, h))
    br = img.crop((w - crop_w, h - crop_h, w, h))
    center = center_crop(img, (crop_h, crop_w))
    return (tl, tr, bl, br, center)


def ten_crop(img, size, vertical_flip=False):
    r"""Crop the given CV Image into four corners and the central crop plus the
        flipped version of these (horizontal flipping is used by default).

    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
       size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
       vertical_flip (bool): Use vertical flipping instead of horizontal

    Returns:
       tuple: tuple (tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip, br_flip, center_flip)
                Corresponding top left, top right, bottom left, bottom right and center crop
                and same for the flipped image.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(
            size) == 2, "Please provide only two dimensions (h, w) for size."

    first_five = five_crop(img, size)

    if vertical_flip:
        img = vflip(img)
    else:
        img = hflip(img)

    second_five = five_crop(img, size)
    return first_five + second_five


def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.

    Args:
        img (CV Image): CV Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        CV Image: Brightness adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    im = img.astype(np.float32) * brightness_factor
    im = im.clip(min=0, max=255)
    return im.astype(img.dtype)


def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.

    Args:
        img (CV Image): CV Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        CV Image: Contrast adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    im = img.astype(np.float32)
    mean = round(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).mean())
    im = (1 - contrast_factor) * mean + contrast_factor * im
    im = im.clip(min=0, max=255)
    return im.astype(img.dtype)


def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.

    Args:
        img (CV Image): CV Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        CV Image: Saturation adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    im = img.astype(np.float32)
    degenerate = cv2.cvtColor(
        cv2.cvtColor(
            im,
            cv2.COLOR_RGB2GRAY),
        cv2.COLOR_GRAY2RGB)
    im = (1 - saturation_factor) * degenerate + saturation_factor * im
    im = im.clip(min=0, max=255)
    return im.astype(img.dtype)


def adjust_hue(img, hue_factor):
    """Adjust hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See `Hue`_ for more details.

    .. _Hue: https://en.wikipedia.org/wiki/Hue

    Args:
        img (CV Image): CV Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        CV Image: Hue adjusted image.
    """
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError(
            'hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    im = img.astype(np.uint8)
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV_FULL)
    hsv[..., 0] += np.uint8(hue_factor * 255)

    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)
    return im.astype(img.dtype)


def adjust_gamma(img, gamma, gain=1):
    r"""Perform gamma correction on an image.

    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:

    .. math::
        I_{\text{out}} = 255 \times \text{gain} \times \left(\frac{I_{\text{in}}}{255}\right)^{\gamma}

    See `Gamma Correction`_ for more details.

    .. _Gamma Correction: https://en.wikipedia.org/wiki/Gamma_correction

    Args:
        img (CV Image): CV Image to be adjusted.
        gamma (float): Non negative real number, same as :math:`\gamma` in the equation.
            gamma larger than 1 make the shadows darker,
            while gamma smaller than 1 make dark regions lighter.
        gain (float): The constant multiplier.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    im = img.astype(np.float32)
    im = 255. * gain * np.power(im / 255., gamma)
    im = im.clip(min=0., max=255.)
    return im.astype(img.dtype)


def rotate(img, angle, resample=False, expand=False, center=None):
    """Rotate the image by angle.


    Args:
        img (CV Image): PIL Image to be rotated.
        angle (float or int): In degrees degrees counter clockwise order.
        resample (``NEAREST`` or ``BILINEAR`` or ``BICUBIC``, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to ``NEAREST``.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    imgtype = img.dtype
    h, w, _ = img.shape
    point = center or (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(point, angle=-angle, scale=1)

    if expand:
        if center is None:
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])

            # compute the new bounding dimensions of the image
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))

            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nW / 2) - point[0]
            M[1, 2] += (nH / 2) - point[1]

            # perform the actual rotation and return the image
            dst = cv2.warpAffine(img, M, (nW, nH))
        else:
            xx = []
            yy = []
            for point in (np.array([0, 0, 1]), np.array(
                    [w - 1, 0, 1]), np.array([w - 1, h - 1, 1]), np.array([0, h - 1, 1])):
                target = M @ point
                xx.append(target[0])
                yy.append(target[1])
            nh = int(math.ceil(max(yy)) - math.floor(min(yy)))
            nw = int(math.ceil(max(xx)) - math.floor(min(xx)))
            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nw - w) / 2
            M[1, 2] += (nh - h) / 2
            dst = cv2.warpAffine(img, M, (nw, nh), flags=INTER_MODE[resample])
    else:
        dst = cv2.warpAffine(img, M, (w, h), flags=INTER_MODE[resample])
    return dst.astype(imgtype)


def _get_inverse_affine_matrix(center, angle, translate, scale, shear):
    # Helper method to compute inverse matrix for affine transformation

    angle = math.radians(angle)
    shear = math.radians(shear)
    M00 = math.cos(angle) * scale
    M01 = -math.sin(angle + shear) * scale
    M10 = math.sin(angle) * scale
    M11 = math.cos(angle + shear) * scale
    M02 = center[0] - center[0] * M00 - center[1] * M01 + translate[0]
    M12 = center[1] - center[0] * M10 - center[1] * M11 + translate[1]
    matrix = np.array([[M00, M01, M02], [M10, M11, M12]], dtype=np.float32)
    return matrix


def affine(
        img,
        angle=0,
        translate=(
            0,
            0),
    scale=1,
    shear=0,
    resample='BILINEAR',
        fillcolor=None):
    """Apply affine transformation on the image keeping image center invariant

    Args:
        img (CV Image): CV Image to be rotated.
        angle (float or int): rotation angle in degrees between -180 and 180, clockwise direction.
        translate (list or tuple of integers): horizontal and vertical translations (post-rotation translation)
        scale (float): overall scale
        shear (float): shear angle value in degrees between -180 to 180, clockwise direction.
        resample (``NEAREST`` or ``BILINEAR`` or ``BICUBIC``, optional):
            An optional resampling filter.
            See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to ``PIL.Image.NEAREST``.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
        "Argument translate should be a list or tuple of length 2"

    assert scale > 0.0, "Argument scale should be positive"
    imgtype = img.dtype
    gray_scale = False
    if len(img.shape) == 2:
        gray_scale = True
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    rows, cols, _ = img.shape
    center = (cols * 0.5, rows * 0.5)
    matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
    dst_img = cv2.warpAffine(
        img,
        matrix,
        (cols,
         rows),
        flags=INTER_MODE[resample],
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fillcolor)
    if gray_scale:
        dst_img = cv2.cvtColor(dst_img, cv2.COLOR_RGB2GRAY)
    return dst_img.astype(imgtype)


def to_grayscale(img, num_output_channels=1):
    """Convert image to grayscale version of image.

    Args:
        img (CV Image): CV to be converted to grayscale.

    Returns:
        CV Image: Grayscale version of the image.
            if num_output_channels = 1 : returned image is single channel

            if num_output_channels = 3 : returned image is 3 channel with r = g = b
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    if num_output_channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif num_output_channels == 3:
        img = cv2.cvtColor(
            cv2.cvtColor(
                img,
                cv2.COLOR_RGB2GRAY),
            cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError('num_output_channels should be either 1 or 3')

    return img


def erase(img, i, j, h, w, v, inplace=False):
    """ Erase the input Tensor Image with given value.

    Args:
        img (Tensor Image): Tensor image of size (C, H, W) to be erased
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the erased region.
        w (int): Width of the erased region.
        v: Erasing value.
        inplace(bool, optional): For in-place operations. By default is set False.

    Returns:
        Tensor Image: Erased image.
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError('img should be Tensor Image. Got {}'.format(type(img)))

    if not inplace:
        img = img.clone()

    img[:, i:i + h, j:j + w] = v
    return img


def cutout(img, i, j, h, w, v, inplace=False):
    """ Erase the CV Image with given value.

    Args:
        img (Tensor Image): Tensor image of size (C, H, W) to be erased
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the erased region.
        w (int): Width of the erased region.
        v: Erasing value.
        inplace(bool, optional): For in-place operations. By default is set False.

    Returns:
        CV Image: Cutout image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    if not inplace:
        img = img.copy()

    img[i:i + h, j:j + w, :] = v
    return img


def gaussian_noise(img: np.ndarray, mean, std):
    imgtype = img.dtype
    gauss = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy = np.clip(gauss + img.astype(np.float32), 0, 255)
    return noisy.astype(imgtype)


def poisson_noise(img):
    imgtype = img.dtype
    img = img.astype(np.float32) / 255.0
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = 255 * \
        np.clip(np.random.poisson(img.astype(np.float32) * vals) / float(vals), 0, 1)
    return noisy.astype(imgtype)


def salt_and_pepper(img, prob=0.01):
    """
    Adds "Salt & Pepper" noise to an image.
    prob: probability (threshold) that controls level of noise
    """

    imgtype = img.dtype
    rnd = np.random.rand(img.shape[0], img.shape[1])
    noisy = img.copy()
    noisy[rnd < prob / 2] = 0.0
    noisy[rnd > 1 - prob / 2] = 255.0
    return noisy.astype(imgtype)


def text_overlay(img, length, font, text_scale):
    assert _is_numpy_image(img) and img.dtype == np.uint8
    h, w, c = img.shape
    length = np.random.randint(*length)
    text_scale = np.random.uniform(*text_scale)
    text = ''.join(random.choice(ascii_letters) for _ in range(length))
    color = np.random.randint(0, 255, c).tolist()
    pos = (random.randint(0, w), random.randint(0, h))
    img = cv2.putText(img, text, pos, font, text_scale, color)
    return img
