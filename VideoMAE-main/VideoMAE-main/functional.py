import numbers
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

def _is_tensor_clip(clip):
    return isinstance(clip, tf.Tensor) and clip.ndim == 4

def crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]

    elif isinstance(clip[0], Image.Image):
        cropped = [
            img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip
        ]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        ' but got list of {0}'.format(type(clip[0])))
    return cropped

def resize_clip(clip, size, interpolation='bilinear'):
    if isinstance(clip[0], np.ndarray):
        if isinstance(size, numbers.Number):
            im_h, im_w, im_c = clip[0].shape
            if (im_w <= im_h and im_w == size) or (im_h <= im_w and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[0], size[1]
        if interpolation == 'bilinear':
            np_inter = cv2.INTER_LINEAR
        else:
            np_inter = cv2.INTER_NEAREST
        scaled = [
            cv2.resize(img, size, interpolation=np_inter) for img in clip
        ]
    elif isinstance(clip[0], Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            pil_inter = Image.BILINEAR
        else:
            pil_inter = Image.NEAREST
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        ' but got list of {0}'.format(type(clip[0])))
    return scaled

def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow

def normalize(clip, mean, std, inplace=False):
    if not _is_tensor_clip(clip):
        raise TypeError('Tensor is not a valid clip.')

    if not inplace:
        clip = tf.identity(clip)

    mean = tf.constant(mean, dtype=clip.dtype)
    std = tf.constant(std, dtype=clip.dtype)
    mean = tf.reshape(mean, (1, -1, 1, 1, 1))
    std = tf.reshape(std, (1, -1, 1, 1, 1))

    clip = (clip - mean) / std
    return clip
