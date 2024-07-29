"""
This implementation is based on
https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/auto_augment.py
pulished under an Apache License 2.0.

COMMENT FROM ORIGINAL:
AutoAugment, RandAugment, and AugMix for PyTorch
This code implements the searched ImageNet policies with various tweaks and
improvements and does not include any of the search code. AA and RA
Implementation adapted from:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py
AugMix adapted from:
    https://github.com/google-research/augmix
Papers:
    AutoAugment: Learning Augmentation Policies from Data
    https://arxiv.org/abs/1805.09501
    Learning Data Augmentation Strategies for Object Detection
    https://arxiv.org/abs/1906.11172
    RandAugment: Practical automated data augmentation...
    https://arxiv.org/abs/1909.13719
    AugMix: A Simple Data Processing Method to Improve Robustness and
    Uncertainty https://arxiv.org/abs/1912.02781

Hacked together by / Copyright 2020 Ross Wightman
"""

import tensorflow as tf
import numpy as np
import random

_MAX_LEVEL = 10.0

_HPARAMS_DEFAULT = {
    "translate_const": 250,
    "img_mean": (128, 128, 128),
}

def _randomly_negate(v):
    """With 50% prob, negate the value"""
    return -v if random.random() > 0.5 else v

def _rotate_level_to_arg(level, _hparams):
    level = (level / _MAX_LEVEL) * 30.0
    level = _randomly_negate(level)
    return (level,)

def _enhance_level_to_arg(level, _hparams):
    return ((level / _MAX_LEVEL) * 1.8 + 0.1,)

def _enhance_increasing_level_to_arg(level, _hparams):
    level = (level / _MAX_LEVEL) * 0.9
    level = 1.0 + _randomly_negate(level)
    return (level,)

def _shear_level_to_arg(level, _hparams):
    level = (level / _MAX_LEVEL) * 0.3
    level = _randomly_negate(level)
    return (level,)

def _translate_abs_level_to_arg(level, hparams):
    translate_const = hparams["translate_const"]
    level = (level / _MAX_LEVEL) * float(translate_const)
    level = _randomly_negate(level)
    return (level,)

def _translate_rel_level_to_arg(level, hparams):
    translate_pct = hparams.get("translate_pct", 0.45)
    level = (level / _MAX_LEVEL) * translate_pct
    level = _randomly_negate(level)
    return (level,)

def _posterize_level_to_arg(level, _hparams):
    return (int((level / _MAX_LEVEL) * 4),)

def _posterize_increasing_level_to_arg(level, hparams):
    return (4 - _posterize_level_to_arg(level, hparams)[0],)

def _posterize_original_level_to_arg(level, _hparams):
    return (int((level / _MAX_LEVEL) * 4) + 4,)

def _solarize_level_to_arg(level, _hparams):
    return (int((level / _MAX_LEVEL) * 256),)

def _solarize_increasing_level_to_arg(level, _hparams):
    return (256 - _solarize_level_to_arg(level, _hparams)[0],)

def _solarize_add_level_to_arg(level, _hparams):
    return (int((level / _MAX_LEVEL) * 110),)


def shear_x(img, factor):
    return tf.keras.preprocessing.image.apply_affine_transform(
        img, shear=factor * 180 / np.pi)

def shear_y(img, factor):
    return tf.keras.preprocessing.image.apply_affine_transform(
        img, shear=factor * 180 / np.pi, channel_axis=-1)

def translate_x_rel(img, pct):
    pixels = pct * tf.shape(img)[1]
    return tf.roll(img, shift=int(pixels), axis=1)

def translate_y_rel(img, pct):
    pixels = pct * tf.shape(img)[0]
    return tf.roll(img, shift=int(pixels), axis=0)

def translate_x_abs(img, pixels):
    return tf.roll(img, shift=int(pixels), axis=1)

def translate_y_abs(img, pixels):
    return tf.roll(img, shift=int(pixels), axis=0)

def rotate(img, degrees):
    return tf.image.rot90(img, k=int(degrees // 90))

def auto_contrast(img):
    return tf.image.adjust_contrast(img, contrast_factor=2)

def invert(img):
    return tf.image.adjust_contrast(img, contrast_factor=-1)

def equalize(img):
    return tf.image.adjust_contrast(img, contrast_factor=1)

def solarize(img, thresh):
    return tf.where(img < thresh, img, 255 - img)

def solarize_add(img, add, thresh=128):
    lut = tf.range(256)
    lut = tf.where(lut < thresh, lut + add, lut)
    lut = tf.clip_by_value(lut, 0, 255)
    return tf.gather(lut, img)

def posterize(img, bits_to_keep):
    shift = 8 - bits_to_keep
    return tf.bitwise.right_shift(tf.bitwise.left_shift(img, shift), shift)

def contrast(img, factor):
    return tf.image.adjust_contrast(img, factor)

def color(img, factor):
    return tf.image.adjust_saturation(img, factor)

def brightness(img, factor):
    return tf.image.adjust_brightness(img, factor)

def sharpness(img, factor):
    return tf.image.adjust_contrast(img, factor)



LEVEL_TO_ARG = {
    "AutoContrast": None,
    "Equalize": None,
    "Invert": None,
    "Rotate": _rotate_level_to_arg,
    "Posterize": _posterize_level_to_arg,
    "PosterizeIncreasing": _posterize_increasing_level_to_arg,
    "PosterizeOriginal": _posterize_original_level_to_arg,
    "Solarize": _solarize_level_to_arg,
    "SolarizeIncreasing": _solarize_increasing_level_to_arg,
    "SolarizeAdd": _solarize_add_level_to_arg,
    "Color": _enhance_level_to_arg,
    "ColorIncreasing": _enhance_increasing_level_to_arg,
    "Contrast": _enhance_level_to_arg,
    "ContrastIncreasing": _enhance_increasing_level_to_arg,
    "Brightness": _enhance_level_to_arg,
    "BrightnessIncreasing": _enhance_increasing_level_to_arg,
    "Sharpness": _enhance_level_to_arg,
    "SharpnessIncreasing": _enhance_increasing_level_to_arg,
    "ShearX": _shear_level_to_arg,
    "ShearY": _shear_level_to_arg,
    "TranslateX": _translate_abs_level_to_arg,
    "TranslateY": _translate_abs_level_to_arg,
    "TranslateXRel": _translate_rel_level_to_arg,
    "TranslateYRel": _translate_rel_level_to_arg,
}

NAME_TO_OP = {
    "AutoContrast": auto_contrast,
    "Equalize": equalize,
    "Invert": invert,
    "Rotate": rotate,
    "Posterize": posterize,
    "PosterizeIncreasing": posterize,
    "PosterizeOriginal": posterize,
    "Solarize": solarize,
    "SolarizeIncreasing": solarize,
    "SolarizeAdd": solarize_add,
    "Color": color,
    "ColorIncreasing": color,
    "Contrast": contrast,
    "ContrastIncreasing": contrast,
    "Brightness": brightness,
    "BrightnessIncreasing": brightness,
    "Sharpness": sharpness,
    "SharpnessIncreasing": sharpness,
    "ShearX": shear_x,
    "ShearY": shear_y,
    "TranslateX": translate_x_abs,
    "TranslateY": translate_y_abs,
    "TranslateXRel": translate_x_rel,
    "TranslateYRel": translate_y_rel,
}


class AugmentOp:
    def __init__(self, name, prob=0.5, magnitude=10, hparams=None):
        hparams = hparams or _HPARAMS_DEFAULT
        self.aug_fn = NAME_TO_OP[name]
        self.level_fn = LEVEL_TO_ARG[name]
        self.prob = prob
        self.magnitude = magnitude
        self.hparams = hparams.copy()

        self.magnitude_std = self.hparams.get("magnitude_std", 0)

    def __call__(self, img):
        if self.prob < 1.0 and random.random() > self.prob:
            return img
        magnitude = self.magnitude
        if self.magnitude_std and self.magnitude_std > 0:
            magnitude = random.gauss(magnitude, self.magnitude_std)
        magnitude = min(_MAX_LEVEL, max(0, magnitude))  # clip to valid range
        level_args = (
            self.level_fn(magnitude, self.hparams)
            if self.level_fn is not None
            else ()
        )

        return self.aug_fn(img, *level_args)




def _select_rand_weights(weight_idx=0, transforms=None):
    transforms = transforms or _RAND_TRANSFORMS
    assert weight_idx == 0  # only one set of weights currently
    rand_weights = _RAND_CHOICE_WEIGHTS_0
    probs = [rand_weights[k] for k in transforms]
    probs /= np.sum(probs)
    return probs

class RandAugment:
    def __init__(self, ops, num_layers=2, choice_weights=None):
        self.ops = ops
        self.num_layers = num_layers
        self.choice_weights = choice_weights

    def distort(self, img):
        op_indices = np.random.choice(
            np.arange(len(self.ops)),
            self.num_layers,
            replace=True,
            p=self.choice_weights,
        )
        for i in op_indices:
            img = self.ops[i](img)
        return img
