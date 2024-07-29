import os
import numpy as np
import tensorflow as tf
from tensorflow import keras 
from keras.preprocessing.image import img_to_array, array_to_img
from decord import VideoReader, cpu
from tensorflow.data import Dataset
import warnings

# Helper functions for normalization and augmentation
def tensor_normalize(tensor, mean, std):
    tensor = tf.image.convert_image_dtype(tensor, dtype=tf.float32)
    tensor = (tensor - mean) / std
    return tensor

def random_resized_crop(images, target_height, target_width, scale, ratio):
    cropped_images = []
    for img in images:
        img = tf.image.random_crop(img, [target_height, target_width, 3])
        img = tf.image.random_flip_left_right(img)
        cropped_images.append(img)
    return tf.stack(cropped_images)

def spatial_sampling(frames, min_scale, max_scale, crop_size, random_horizontal_flip=True, aspect_ratio=None, scale=None):
    if aspect_ratio is None and scale is None:
        frames = tf.image.resize(frames, [min_scale, max_scale])
        frames = tf.image.random_crop(frames, [frames.shape[0], crop_size, crop_size, 3])
    else:
        frames = random_resized_crop(frames, crop_size, crop_size, scale, aspect_ratio)
    if random_horizontal_flip:
        frames = tf.image.random_flip_left_right(frames)
    return frames

class SSVideoClsDataset(Dataset):
    def __init__(self, anno_path, data_path, mode='train', clip_len=8, crop_size=224, short_side_size=256, new_height=256,
                 new_width=340, keep_aspect_ratio=True, num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3, args=None):
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False
        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True

        import pandas as pd
        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=' ')
        self.dataset_samples = list(cleaned.values[:, 0])
        self.label_array = list(cleaned.values[:, 1])

        if self.mode == 'train':
            pass

        elif self.mode == 'validation':
            self.data_transform = tf.keras.Sequential([
                tf.keras.layers.Resizing(self.short_side_size, self.short_side_size),
                tf.keras.layers.CenterCrop(self.crop_size, self.crop_size),
                tf.keras.layers.Rescaling(scale=1./255),
                tf.keras.layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229, 0.224, 0.225])
            ])
        elif self.mode == 'test':
            self.data_resize = tf.keras.Sequential([
                tf.keras.layers.Resizing(self.short_side_size, self.short_side_size)
            ])
            self.data_transform = tf.keras.Sequential([
                tf.keras.layers.Rescaling(scale=1./255),
                tf.keras.layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229, 0.224, 0.225])
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        sample_label = self.label_array[idx]
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_seg.append((ck, cp))

    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args
            scale_t = 1

            sample = self.dataset_samples[index]
            buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t)

            if args.num_sample > 1:
                frame_list = []
                label_list = []
                index_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self.label_array[index]
                    frame_list.append(new_frames)
                    label_list.append(label)
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                buffer = self._aug_frame(buffer, args)

            return buffer, self.label_array[index], index, {}

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            buffer = self.loadvideo_decord(sample)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample)
            buffer = self.data_transform(buffer)
            return buffer, self.label_array[index], sample.split("/")[-1].split(".")[0]

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.loadvideo_decord(sample)

            while len(buffer) == 0:
                warnings.warn("video {}, temporal {}, spatial {} not found during testing".format(
                    str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer = self.loadvideo_decord(sample)

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                           / (self.test_num_crop - 1)
            temporal_start = chunk_nb  # 0/1
            spatial_start = int(split_nb * spatial_step)
            if buffer.shape[1] >= buffer.shape[2]:
                buffer = buffer[temporal_start::2,
                         spatial_start:spatial_start + self.short_side_size, :, :]
            else:
                buffer = buffer[temporal_start::2,
                         :, spatial_start:spatial_start + self.short_side_size, :]

            buffer = self.data_transform(buffer)
            return buffer, self.test_label_array[index], sample.split("/")[-1].split(".")[0], \
                   chunk_nb, split_nb
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(self, buffer, args):
        aug_transform = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomCrop(self.crop_size, self.crop_size),
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
            tf.keras.layers.experimental.preprocessing.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229, 0.224, 0.225])
        ])

        buffer = [tf.keras.preprocessing.image.array_to_img(frame) for frame in buffer]
        buffer = aug_transform(buffer)
        buffer = [tf.keras.preprocessing.image.img_to_array(img) for img in buffer]
        buffer = tf.stack(buffer)
        buffer = tf.transpose(buffer, perm=[0, 2, 3, 1])

        buffer = tensor_normalize(buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        buffer = tf.transpose(buffer, perm=[3, 0, 1, 2])

        buffer = spatial_sampling(buffer, min_scale=256, max_scale=320, crop_size=self.crop_size,
                                  random_horizontal_flip=False if args.data_set == 'SSV2' else True,
                                  aspect_ratio=[0.75, 1.3333], scale=[0.08, 1.0], motion_shift=False)

        return buffer

    def loadvideo_decord(self, sample, sample_rate_scale=1):
        fname = sample

        if not os.path.exists(fname):
            return []

        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []
        try:
            if self.keep_aspect_ratio:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(fname, width=self.new_width, height=self.new_height,
                                 num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

        if self.mode == 'test':
            all_index = []
            tick = len(vr) / float(self.num_segment)
            all_index = list(np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segment)] +
                                      [int(tick * x) for x in range(self.num_segment)]))
            while len(all_index) < (self.num_segment * self.test_num_segment):
                all_index.append(all_index[-1])
            all_index = list(np.sort(np.array(all_index)))
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer

        average_duration = len(vr) // self.num_segment
        all_index = []
        if average_duration > 0:
            all_index += list(np.multiply(list(range(self.num_segment)), average_duration) + np.random.randint(average_duration,
                                                                                                               size=self.num_segment))
        elif len(vr) > self.num_segment:
            all_index += list(np.sort(np.random.randint(len(vr), size=self.num_segment)))
        else:
            all_index += list(np.zeros((self.num_segment,)))
        all_index = list(np.array(all_index))
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)
