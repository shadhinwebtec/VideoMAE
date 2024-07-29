import os
import numpy as np
import tensorflow as tf
from decord import VideoReader, cpu
import pandas as pd

class VideoClsDataset(tf.data.Dataset):
    

    def __new__(cls, anno_path, data_path, mode='train', clip_len=8,
                frame_sample_rate=2, crop_size=224, short_side_size=256,
                new_height=256, new_width=340, keep_aspect_ratio=True,
                num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3, args=None):
        self = tf.data.Dataset.__new__(cls)
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
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
        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")

        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=' ')
        self.dataset_samples = list(cleaned.values[:, 0])
        self.label_array = list(cleaned.values[:, 1])

        self.data_transform = self.get_data_transform()

        data = list(zip(self.dataset_samples, self.label_array))
        self.data = tf.data.Dataset.from_tensor_slices(data)
        self.data = self.data.map(self.process_sample, num_parallel_calls=tf.data.AUTOTUNE)
        
        return self.data

    def get_data_transform(self):
        if self.mode == 'validation':
            return tf.keras.Sequential([
                tf.keras.layers.experimental.preprocessing.Resizing(self.short_side_size, self.short_side_size),
                tf.keras.layers.CenterCrop(self.crop_size, self.crop_size),
                tf.keras.layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229, 0.224, 0.225])
            ])
        elif self.mode == 'test':
            return tf.keras.Sequential([
                tf.keras.layers.experimental.preprocessing.Resizing(self.short_side_size, self.short_side_size),
                tf.keras.layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229, 0.224, 0.225])
            ])
        else:
            return None

    def process_sample(self, sample, label):
        if self.mode == 'train':
            buffer = self.load_video(sample)
            if self.args.num_sample > 1:
                frame_list = []
                label_list = []
                for _ in range(self.args.num_sample):
                    new_frames = self.augment_frames(buffer)
                    frame_list.append(new_frames)
                    label_list.append(label)
                return tf.stack(frame_list), tf.convert_to_tensor(label_list)
            else:
                buffer = self.augment_frames(buffer)
            return buffer, label

        buffer = self.load_video(sample)
        buffer = self.data_transform(buffer)
        return buffer, label

    def load_video(self, sample):
        fname = sample.numpy().decode("utf-8")

        if not os.path.exists(fname):
            return tf.convert_to_tensor([])

        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return tf.convert_to_tensor([])

        try:
            if self.keep_aspect_ratio:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(fname, width=self.new_width, height=self.new_height, num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            return tf.convert_to_tensor([])

        if self.mode == 'test':
            all_index = [x for x in range(0, len(vr), self.frame_sample_rate)]
            while len(all_index) < self.clip_len:
                all_index.append(all_index[-1])
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return tf.convert_to_tensor(buffer)

        converted_len = int(self.clip_len * self.frame_sample_rate)
        seg_len = len(vr) // self.num_segment

        all_index = []
        for i in range(self.num_segment):
            if seg_len <= converted_len:
                index = np.linspace(0, seg_len, num=seg_len // self.frame_sample_rate)
                index = np.concatenate((index, np.ones(self.clip_len - seg_len // self.frame_sample_rate) * seg_len))
                index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                end_idx = np.random.randint(converted_len, seg_len)
                str_idx = end_idx - converted_len
                index = np.linspace(str_idx, end_idx, num=self.clip_len)
                index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
            index = index + i * seg_len
            all_index.extend(list(index))

        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return tf.convert_to_tensor(buffer)

    def augment_frames(self, buffer):
        # Implement the augmentation logic using tf.image
        buffer = [tf.image.convert_image_dtype(frame, tf.float32) for frame in buffer]
        buffer = [tf.image.resize_with_crop_or_pad(frame, self.crop_size, self.crop_size) for frame in buffer]
        buffer = [tf.image.random_flip_left_right(frame) for frame in buffer]
        buffer = tf.stack(buffer)
        buffer = tf.image.per_image_standardization(buffer)
        return buffer


class VideoMAE(tf.data.Dataset):
    

    def __new__(cls, root, setting, train=True, test_mode=False, name_pattern='img_%05d.jpg', video_ext='mp4',
                is_color=True, modality='rgb', num_segments=1, num_crop=1, new_length=1, new_step=1, transform=None,
                temporal_jitter=False, video_loader=False, use_decord=True, lazy_init=False):
        self = tf.data.Dataset.__new__(cls)
        self.root = root
        self.setting = setting
        self.train = train
        self.test_mode = test_mode
        self.is_color = is_color
        self.modality = modality
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.temporal_jitter = temporal_jitter
        self.name_pattern = name_pattern
        self.video_loader = video_loader
        self.video_ext = video_ext
        self.use_decord = use_decord
        self.transform = transform
        self.lazy_init = lazy_init

        if not self.lazy_init:
            self.clips = self._make_dataset(root, setting)
            if len(self.clips) == 0:
                raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                                   "Check your data directory (opt.data-dir)."))

        data = list(zip(self.clips, [i for i in range(len(self.clips))]))
        self.data = tf.data.Dataset.from_tensor_slices(data)
        self.data = self.data.map(self.process_sample, num_parallel_calls=tf.data.AUTOTUNE)

        return self.data

    def process_sample(self, clip, index):
        directory, target = clip

        if self.video_loader:
            video_name = '{}.{}'.format(directory, self.video_ext)
            decord_vr = VideoReader(video_name, num_threads=1)
            duration = len(decord_vr)

        segment_indices, skip_offsets = self._sample_train_indices(duration)
        images = self._video_TSN_decord_batch_loader(directory, decord_vr, duration, segment_indices, skip_offsets)

        process_data, mask = self.transform((images, None))  # T*C,H,W
        process_data = tf.reshape(process_data, (self.new_length, 3) + process_data.shape[-2:]).transpose(0, 1)  # T*C,H,W -> T,C,H,W -> C,T,H,W

        return process_data, mask

    def _make_dataset(self, directory, setting):
        if not os.path.exists(setting):
            raise(RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (setting)))
        clips = []
        with open(setting) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split(' ')
                clip_path = os.path.join(line_info[0])
                target = int(line_info[1])
                item = (clip_path, target)
                clips.append(item)
        return clips

    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + \
                      np.random.randint(average_duration, size=self.num_segments)
            offsets = offsets.tolist()
        elif num_frames > self.num_segments:
            offsets = np.sort(np.random.randint(num_frames - self.skip_length + 1, size=self.num_segments))
            offsets = offsets.tolist()
        else:
            offsets = np.zeros((self.num_segments,))
            offsets = offsets.tolist()
        return offsets, [0] * self.num_segments

    def _video_TSN_decord_batch_loader(self, directory, video_reader, duration, indices, skip_offsets):
        frame_indices = []
        for seg_ind in range(self.num_segments):
            p = int(indices[seg_ind])
            for i in range(self.new_length):
                if p + skip_offsets[seg_ind] < duration:
                    frame_indices.append(p + skip_offsets[seg_ind])
                if p + skip_offsets[seg_ind] < duration - 1:
                    p += self.new_step

        return video_reader.get_batch(frame_indices).asnumpy()
