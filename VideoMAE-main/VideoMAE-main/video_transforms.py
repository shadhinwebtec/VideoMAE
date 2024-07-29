#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import math
from PIL import Image

def _pil_interp(method):
    if method == "bicubic":
        return Image.BICUBIC
    elif method == "lanczos":
        return Image.LANCZOS
    elif method == "hamming":
        return Image.HAMMING
    else:
        return Image.BILINEAR

def random_short_side_scale_jitter(images, min_size, max_size, boxes=None, inverse_uniform_sampling=False):
    if inverse_uniform_sampling:
        size = int(round(1.0 / np.random.uniform(1.0 / max_size, 1.0 / min_size)))
    else:
        size = int(round(np.random.uniform(min_size, max_size)))

    height = tf.shape(images)[1]
    width = tf.shape(images)[2]
    
    if (width <= height and width == size) or (height <= width and height == size):
        return images, boxes
    
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
        if boxes is not None:
            boxes = boxes * float(new_height) / height
    else:
        new_width = int(math.floor((float(width) / height) * size))
        if boxes is not None:
            boxes = boxes * float(new_width) / width

    images = tf.image.resize(images, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR)
    return images, boxes

def crop_boxes(boxes, x_offset, y_offset):
    cropped_boxes = boxes.copy()
    cropped_boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
    cropped_boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset
    return cropped_boxes

def random_crop(images, size, boxes=None):
    height = tf.shape(images)[1]
    width = tf.shape(images)[2]
    
    if height == size and width == size:
        return images, boxes
    
    y_offset = 0
    if height > size:
        y_offset = int(np.random.randint(0, height - size))
    
    x_offset = 0
    if width > size:
        x_offset = int(np.random.randint(0, width - size))
    
    cropped = images[:, y_offset:y_offset+size, x_offset:x_offset+size, :]
    cropped_boxes = crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None
    return cropped, cropped_boxes

def horizontal_flip(prob, images, boxes=None):
    if boxes is None:
        flipped_boxes = None
    else:
        flipped_boxes = boxes.copy()

    if np.random.uniform() < prob:
        images = tf.image.flip_left_right(images)

        if boxes is not None:
            width = tf.shape(images)[2]
            flipped_boxes[:, [0, 2]] = width - boxes[:, [2, 0]] - 1

    return images, flipped_boxes

def uniform_crop(images, size, spatial_idx, boxes=None, scale_size=None):
    assert spatial_idx in [0, 1, 2]
    ndim = len(images.shape)
    
    if ndim == 3:
        images = tf.expand_dims(images, axis=0)
    
    height = tf.shape(images)[1]
    width = tf.shape(images)[2]

    if scale_size is not None:
        if width <= height:
            width, height = scale_size, int(height / width * scale_size)
        else:
            width, height = int(width / height * scale_size), scale_size
        images = tf.image.resize(images, [height, width], method=tf.image.ResizeMethod.BILINEAR)

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    
    cropped = images[:, y_offset:y_offset+size, x_offset:x_offset+size, :]
    cropped_boxes = crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None
    
    if ndim == 3:
        cropped = tf.squeeze(cropped, axis=0)
    
    return cropped, cropped_boxes




def clip_boxes_to_image(boxes, height, width):
    """
    Clip an array of boxes to an image with the given height and width.
    Args:
        boxes (ndarray): bounding boxes to perform clipping.
            Dimension is `num boxes` x 4.
        height (int): given image height.
        width (int): given image width.
    Returns:
        clipped_boxes (ndarray): the clipped boxes with dimension of
            `num boxes` x 4.
    """
    clipped_boxes = np.copy(boxes)
    clipped_boxes[:, [0, 2]] = np.minimum(
        width - 1.0, np.maximum(0.0, boxes[:, [0, 2]])
    )
    clipped_boxes[:, [1, 3]] = np.minimum(
        height - 1.0, np.maximum(0.0, boxes[:, [1, 3]])
    )
    return clipped_boxes

def blend(images1, images2, alpha):
    """
    Blend two images with a given weight alpha.
    Args:
        images1 (tensor): the first images to be blended, the dimension is
            `num frames` x `height` x `width` x `channel`.
        images2 (tensor): the second images to be blended, the dimension is
            `num frames` x `height` x `width` x `channel`.
        alpha (float): the blending weight.
    Returns:
        (tensor): blended images, the dimension is
            `num frames` x `height` x `width` x `channel`.
    """
    return images1 * alpha + images2 * (1 - alpha)

def grayscale(images):
    """
    Get the grayscale for the input images. The channels of images should be
    in order BGR.
    Args:
        images (tensor): the input images for getting grayscale. Dimension is
            `num frames` x `height` x `width` x `channel`.
    Returns:
        img_gray (tensor): grayscale images, the dimension is
            `num frames` x `height` x `width` x `channel`.
    """
    # R -> 0.299, G -> 0.587, B -> 0.114.
    r, g, b = tf.split(images, 3, axis=-1)
    gray_channel = 0.299 * b + 0.587 * g + 0.114 * r
    img_gray = tf.concat([gray_channel] * 3, axis=-1)
    return img_gray

def color_jitter(images, img_brightness=0, img_contrast=0, img_saturation=0):
    """
    Perform a color jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `height` x `width` x `channel`.
        img_brightness (float): jitter ratio for brightness.
        img_contrast (float): jitter ratio for contrast.
        img_saturation (float): jitter ratio for saturation.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `height` x `width` x `channel`.
    """
    jitter = []
    if img_brightness != 0:
        jitter.append("brightness")
    if img_contrast != 0:
        jitter.append("contrast")
    if img_saturation != 0:
        jitter.append("saturation")

    if len(jitter) > 0:
        order = np.random.permutation(np.arange(len(jitter)))
        for idx in range(len(jitter)):
            if jitter[order[idx]] == "brightness":
                images = brightness_jitter(img_brightness, images)
            elif jitter[order[idx]] == "contrast":
                images = contrast_jitter(img_contrast, images)
            elif jitter[order[idx]] == "saturation":
                images = saturation_jitter(img_saturation, images)
    return images

def brightness_jitter(var, images):
    """
    Perform brightness jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        var (float): jitter ratio for brightness.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `height` x `width` x `channel`.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `height` x `width` x `channel`.
    """
    alpha = 1.0 + np.random.uniform(-var, var)
    img_bright = tf.zeros_like(images)
    images = blend(images, img_bright, alpha)
    return images

def contrast_jitter(var, images):
    """
    Perform contrast jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        var (float): jitter ratio for contrast.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `height` x `width` x `channel`.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `height` x `width` x `channel`.
    """
    alpha = 1.0 + np.random.uniform(-var, var)
    img_gray = grayscale(images)
    mean_gray = tf.reduce_mean(img_gray, axis=(1, 2), keepdims=True)
    img_gray = tf.tile(mean_gray, [1, tf.shape(images)[1], tf.shape(images)[2], 1])
    images = blend(images, img_gray, alpha)
    return images

def saturation_jitter(var, images):
    """
    Perform saturation jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        var (float): jitter ratio for saturation.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `height` x `width` x `channel`.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `height` x `width` x `channel`.
    """
    alpha = 1.0 + np.random.uniform(-var, var)
    img_gray = grayscale(images)
    images = blend(images, img_gray, alpha)
    return images

def lighting_jitter(images, alphastd, eigval, eigvec):
    """
    Perform AlexNet-style PCA jitter on the given images.
    Args:
        images (tensor): images to perform lighting jitter. Dimension is
            `num frames` x `height` x `width` x `channel`.
        alphastd (float): jitter ratio for PCA jitter.
        eigval (list): eigenvalues for PCA jitter.
        eigvec (list[list]): eigenvectors for PCA jitter.
    Returns:
        out_images (tensor): the jittered images, the dimension is
            `num frames` x `height` x `width` x `channel`.
    """
    if alphastd == 0:
        return images

    # Generate alpha1, alpha2, alpha3
    alpha = np.random.normal(0, alphastd, size=(1, 3))
    eig_vec = np.array(eigvec)
    eig_val = np.reshape(eigval, (1, 3))
    rgb = np.sum(
        eig_vec * np.repeat(alpha, 3, axis=0) * np.repeat(eig_val, 3, axis=0),
        axis=1,
    )
    out_images = tf.zeros_like(images)
    
    if len(images.shape) == 3:
        # C H W
        channel_dim = 0
    elif len(images.shape) == 4:
        # T C H W
        channel_dim = 1
    else:
        raise NotImplementedError(f"Unsupported dimension {len(images.shape)}")

    for idx in range(images.shape[channel_dim]):
        if len(images.shape) == 3:
            out_images[idx] = images[idx] + rgb[2 - idx]
        elif len(images.shape) == 4:
            out_images[:, idx] = images[:, idx] + rgb[2 - idx]
        else:
            raise NotImplementedError(f"Unsupported dimension {len(images.shape)}")

    return out_images


import tensorflow as tf

def color_normalization(images, mean, stddev):
    """
    Perform color normalization on the given images.
    Args:
        images (tensor): images to perform color normalization. Dimension is
            `num frames` x `channel` x `height` x `width`.
        mean (list): mean values for normalization.
        stddev (list): standard deviations for normalization.

    Returns:
        out_images (tensor): the normalized images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    if len(images.shape) == 3:
        assert len(mean) == images.shape[0], "channel mean not computed properly"
        assert len(stddev) == images.shape[0], "channel stddev not computed properly"
    elif len(images.shape) == 4:
        assert len(mean) == images.shape[1], "channel mean not computed properly"
        assert len(stddev) == images.shape[1], "channel stddev not computed properly"
    else:
        raise NotImplementedError(f"Unsupported dimension {len(images.shape)}")

    mean = tf.constant(mean, dtype=images.dtype)
    stddev = tf.constant(stddev, dtype=images.dtype)

    if len(images.shape) == 3:
        out_images = (images - mean[:, tf.newaxis, tf.newaxis]) / stddev[:, tf.newaxis, tf.newaxis]
    elif len(images.shape) == 4:
        out_images = (images - mean[tf.newaxis, :, tf.newaxis, tf.newaxis]) / stddev[tf.newaxis, :, tf.newaxis, tf.newaxis]

    return out_images




def _get_param_spatial_crop(scale, ratio, height, width, num_repeat=10, log_scale=True, switch_hw=False):
    """
    Given scale, ratio, height and width, return sampled coordinates of the videos.
    """
    for _ in range(num_repeat):
        area = height * width
        target_area = random.uniform(*scale) * area
        if log_scale:
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))
        else:
            aspect_ratio = random.uniform(*ratio)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if np.random.uniform() < 0.5 and switch_hw:
            w, h = h, w

        if 0 < w <= width and 0 < h <= height:
            i = random.randint(0, height - h)
            j = random.randint(0, width - w)
            return i, j, h, w

    # Fallback to central crop
    in_ratio = width / height
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:  # whole image
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w




def random_resized_crop(images, target_height, target_width, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
    """
    Crop the given images to random size and aspect ratio. The crop is finally resized to the given size.
    """
    height = images.shape[1]
    width = images.shape[2]

    i, j, h, w = _get_param_spatial_crop(scale, ratio, height, width)
    cropped = images[:, i:i + h, j:j + w, :]
    cropped = tf.image.resize(cropped, [target_height, target_width], method='bilinear')
    return cropped



def random_resized_crop_with_shift(images, target_height, target_width, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
    """
    Samples two different boxes (for cropping) for the first and last frame.
    """
    t = images.shape[0]
    height = images.shape[1]
    width = images.shape[2]

    i, j, h, w = _get_param_spatial_crop(scale, ratio, height, width)
    i_, j_, h_, w_ = _get_param_spatial_crop(scale, ratio, height, width)
    
    i_s = np.linspace(i, i_, num=t, dtype=int)
    j_s = np.linspace(j, j_, num=t, dtype=int)
    h_s = np.linspace(h, h_, num=t, dtype=int)
    w_s = np.linspace(w, w_, num=t, dtype=int)

    out_images = []
    for ind in range(t):
        cropped = images[ind, i_s[ind]:i_s[ind] + h_s[ind], j_s[ind]:j_s[ind] + w_s[ind], :]
        resized = tf.image.resize(cropped[tf.newaxis, :, :, :], [target_height, target_width], method='bilinear')[0]
        out_images.append(resized)

    return tf.stack(out_images)



def random_sized_crop_img(im, size, jitter_scale=(0.08, 1.0), jitter_aspect=(3.0 / 4.0, 4.0 / 3.0), max_iter=10):
    """
    Performs Inception-style cropping (used for training).
    """
    assert len(im.shape) == 3, "Currently only support image for random_sized_crop"
    h, w = im.shape[0:2]
    
    i, j, h, w = _get_param_spatial_crop(
        scale=jitter_scale,
        ratio=jitter_aspect,
        height=h,
        width=w,
        num_repeat=max_iter,
        log_scale=False,
        switch_hw=True,
    )
    
    cropped = im[i:i + h, j:j + w, :]
    resized = tf.image.resize(cropped[tf.newaxis, :, :, :], [size, size], method='bilinear')[0]
    return resized


import tensorflow as tf
import numpy as np
import random
import math

class RandomResizedCropAndInterpolation(tf.keras.layers.Layer):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation='bilinear'):
        super(RandomResizedCropAndInterpolation, self).__init__()
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def call(self, images):
        height, width = tf.shape(images)[1:3]
        area = height * width
        target_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = math.exp(random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))
        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if tf.random.uniform([]) < 0.5:
            w, h = h, w

        if w <= width and h <= height:
            i = random.randint(0, height - h)
            j = random.randint(0, width - w)
        else:
            i, j, h, w = 0, 0, height, width

        cropped = tf.image.crop_to_bounding_box(images, i, j, h, w)
        resized = tf.image.resize(cropped, [self.size, self.size], method=self.interpolation)
        return resized


class RandomHorizontalFlip(tf.keras.layers.Layer):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlip, self).__init__()
        self.p = p

    def call(self, images):
        if tf.random.uniform([]) < self.p:
            images = tf.image.flip_left_right(images)
        return images


class RandomVerticalFlip(tf.keras.layers.Layer):
    def __init__(self, p=0.0):
        super(RandomVerticalFlip, self).__init__()
        self.p = p

    def call(self, images):
        if tf.random.uniform([]) < self.p:
            images = tf.image.flip_up_down(images)
        return images


class RandomRotation(tf.keras.layers.Layer):
    def __init__(self, degrees):
        super(RandomRotation, self).__init__()
        if isinstance(degrees, (tuple, list)):
            self.degrees = degrees
        else:
            self.degrees = (-degrees, degrees)

    def call(self, images):
        angle = tf.random.uniform([], minval=self.degrees[0], maxval=self.degrees[1])
        angle = tf.convert_to_tensor(angle, dtype=tf.float32)
        return tf.image.rotate(images, angle)


class ColorJitter(tf.keras.layers.Layer):
    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0):
        super(ColorJitter, self).__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def call(self, images):
        images = tf.image.random_brightness(images, max_delta=self.brightness)
        images = tf.image.random_contrast(images, lower=1 - self.contrast, upper=1 + self.contrast)
        images = tf.image.random_saturation(images, lower=1 - self.saturation, upper=1 + self.saturation)
        images = tf.image.random_hue(images, max_delta=self.hue)
        return images


class RandomErasing(tf.keras.layers.Layer):
    def __init__(self, prob=0.0, mode="const", max_count=1, num_splits=0, device="cpu", cube=False):
        super(RandomErasing, self).__init__()
        self.prob = prob
        self.mode = mode
        self.max_count = max_count
        self.num_splits = num_splits
        self.device = device
        self.cube = cube

    def call(self, images):
        # TensorFlow implementation of Random Erasing is not available by default.
        # You may need to implement this from scratch based on the specifics of the method.
        return images


def transforms_imagenet_train(img_size=224, scale=None, ratio=None, hflip=0.5, vflip=0.0, color_jitter=0.4,
                               auto_augment=None, interpolation="bilinear", use_prefetcher=False, mean=(0.485, 0.456, 0.406),
                               std=(0.229, 0.224, 0.225), re_prob=0.0, re_mode="const", re_count=1, re_num_splits=0,
                               separate=False):
    scale = scale or (0.08, 1.0)
    ratio = ratio or (3.0 / 4.0, 4.0 / 3.0)

    primary_tfl = [
        RandomResizedCropAndInterpolation(img_size, scale=scale, ratio=ratio, interpolation=interpolation)
    ]
    if hflip > 0.0:
        primary_tfl.append(RandomHorizontalFlip(p=hflip))
    if vflip > 0.0:
        primary_tfl.append(RandomVerticalFlip(p=vflip))

    secondary_tfl = []
    if auto_augment:
        raise NotImplementedError("Auto augment not implemented in TensorFlow")
    elif color_jitter:
        color_jitter_args = (color_jitter,) * 4 if isinstance(color_jitter, (int, float)) else color_jitter
        secondary_tfl.append(ColorJitter(*color_jitter_args))

    final_tfl = [
        tf.keras.layers.Rescaling(scale=1.0 / 255.0),  # Rescale to [0, 1]
        tf.keras.layers.Rescaling(scale=1.0 / np.array(std), offset=-np.array(mean) / np.array(std)),  # Normalize
    ]
    if re_prob > 0.0:
        final_tfl.append(RandomErasing(re_prob, re_mode, re_count, re_num_splits, "cpu", False))

    if separate:
        return tf.keras.Sequential(primary_tfl), tf.keras.Sequential(secondary_tfl), tf.keras.Sequential(final_tfl)
    else:
        return tf.keras.Sequential(primary_tfl + secondary_tfl + final_tfl)



class ColorJitter(tf.keras.layers.Layer):
    """Randomly change the brightness, contrast, saturation, and hue of the images.
    Args:
    brightness (float): How much to jitter brightness.
    contrast (float): How much to jitter contrast.
    saturation (float): How much to jitter saturation.
    hue (float): How much to jitter hue.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super(ColorJitter, self).__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self):
        brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness) if self.brightness > 0 else None
        contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast) if self.contrast > 0 else None
        saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation) if self.saturation > 0 else None
        hue_factor = random.uniform(-self.hue, self.hue) if self.hue > 0 else None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def call(self, images):
        brightness, contrast, saturation, hue = self.get_params()
        
        if brightness is not None:
            images = tf.image.random_brightness(images, max_delta=brightness - 1)
        if contrast is not None:
            images = tf.image.random_contrast(images, lower=1 - contrast, upper=1 + contrast)
        if saturation is not None:
            images = tf.image.random_saturation(images, lower=1 - saturation, upper=1 + saturation)
        if hue is not None:
            images = tf.image.random_hue(images, max_delta=hue)
        
        # Ensure image values are in [0, 1]
        images = tf.clip_by_value(images, 0.0, 1.0)
        return images

class Normalize(tf.keras.layers.Layer):
    """Normalize images with mean and standard deviation.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def call(self, images):
        # Assuming images are normalized to [0, 1] range
        images = tf.image.per_image_standardization(images)
        images = (images - self.mean) / self.std
        return images

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean.tolist()}, std={self.std.tolist()})"
