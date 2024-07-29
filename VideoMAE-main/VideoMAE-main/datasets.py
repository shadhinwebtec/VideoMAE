import os
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import resource

class DataAugmentationForVideoMAE:
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        self.input_size = args.input_size
        
        self.train_augmentation = keras.Sequential([
            tf.keras.layers.RandomCrop(self.input_size, self.input_size),
            tf.keras.layers.RandomFlip("horizontal")
        ])

        self.normalize = lambda x: (x - self.input_mean) / self.input_std
        
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data = self.train_augmentation(images)
        process_data = self.normalize(process_data)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.train_augmentation)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    
    def load_video(path):
        video = tf.io.read_file(path)
        video = tf.io.decode_jpeg(video)
        return transform(video)

    dataset = tf.data.Dataset.list_files(args.data_path + '/*.mp4')
    dataset = dataset.map(load_video, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    print("Data Aug = %s" % str(transform))
    return dataset


def build_dataset(is_train, test_mode, args):
    def parse_function(filename, label):
        video = tf.io.read_file(filename)
        video = tf.io.decode_jpeg(video)
        video = tf.image.resize(video, [256, 320])
        video = tf.image.random_crop(video, [args.num_frames, args.input_size, args.input_size, 3])
        video = (video - tf.constant(self.input_mean)) / tf.constant(self.input_std)
        return video, label

    if args.data_set == 'Kinetics-400':
        dataset_name = 'kinetics400'
        nb_classes = 400
    elif args.data_set == 'SSV2':
        dataset_name = 'something_something_v2'
        nb_classes = 174
    elif args.data_set == 'UCF101':
        dataset_name = 'ucf101'
        nb_classes = 101
    elif args.data_set == 'HMDB51':
        dataset_name = 'hmdb51'
        nb_classes = 51
    else:
        raise NotImplementedError()
    
    split = 'train' if is_train else 'test' if test_mode else 'validation'
    dataset = tfds.load(dataset_name, split=split)
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
