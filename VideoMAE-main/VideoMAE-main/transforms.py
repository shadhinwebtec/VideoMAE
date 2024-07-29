import tensorflow as tf
import random
import numpy as np
import numbers
from PIL import Image

class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        
        w, h = img_group[0].shape[:2]
        th, tw = self.size

        out_images = []

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.shape[0] == w and img.shape[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img[y1:y1+th, x1:x1+tw])

        return (out_images, label)

class GroupCenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        out_images = [tf.image.central_crop(img, self.size / min(img.shape[:2])) for img in img_group]
        return (out_images, label)

class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor_tuple):
        tensor, label = tensor_tuple
        mean = tf.constant(self.mean, dtype=tf.float32)
        std = tf.constant(self.std, dtype=tf.float32)
        tensor = (tensor - mean) / std
        return (tensor, label)

class GroupGrayScale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        out_images = [tf.image.rgb_to_grayscale(img) for img in img_group]
        return (out_images, label)

class GroupScale(object):
    """ Rescales the input image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: tf.image.ResizeMethod.BILINEAR
    """
    def __init__(self, size, interpolation=tf.image.ResizeMethod.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        out_images = [tf.image.resize(img, (self.size, self.size), method=self.interpolation) for img in img_group]
        return (out_images, label)

class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = tf.image.ResizeMethod.BILINEAR

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        
        im_size = img_group[0].shape[:2]

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img[offset_h:offset_h+crop_h, offset_w:offset_w+crop_w] for img in img_group]
        ret_img_group = [tf.image.resize(img, (self.input_size[0], self.input_size[1]), method=self.interpolation) for img in crop_img_group]
        return (ret_img_group, label)

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower right quarter
        return ret

class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        
        if img_group[0].mode == 'L':
            return (np.concatenate([np.expand_dims(np.array(x), 2) for x in img_group], axis=2), label)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return (np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2), label)
            else:
                return (np.concatenate([np.array(x) for x in img_group], axis=2), label)

class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a tensorflow.Tensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic_tuple):
        pic, label = pic_tuple
        
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = tf.convert_to_tensor(pic)
            img = tf.transpose(img, perm=[2, 0, 1])
        else:
            # handle PIL Image
            img = tf.convert_to_tensor(np.array(pic), dtype=tf.float32)
            img = tf.transpose(img, perm=[2, 0, 1])
        return (img / 255.0 if self.div else img, label)

class IdentityTransform(object):

    def __call__(self, data):
        return data
