import math
import sys
from typing import Iterable
import tensorflow as tf
import numpy as np
from einops import rearrange
import utils

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

def train_one_epoch(model: tf.keras.Model, data_loader: Iterable, optimizer: tf.keras.optimizers.Optimizer,
                    epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = tf.keras.losses.MeanSquaredError()

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            if lr_schedule_values is not None:
                optimizer.learning_rate.assign(lr_schedule_values[it] * optimizer.lr)
            if wd_schedule_values is not None and optimizer.weight_decay > 0:
                optimizer.weight_decay.assign(wd_schedule_values[it])

        videos, bool_masked_pos = batch
        bool_masked_pos = tf.reshape(bool_masked_pos, (bool_masked_pos.shape[0], -1))

        mean = tf.constant(IMAGENET_DEFAULT_MEAN, dtype=tf.float32)[None, :, None, None, None]
        std = tf.constant(IMAGENET_DEFAULT_STD, dtype=tf.float32)[None, :, None, None, None]
        unnorm_videos = videos * std + mean  # in [0, 1]

        if normlize_target:
            videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
            videos_norm = (videos_squeeze - tf.reduce_mean(videos_squeeze, axis=-2, keepdims=True)
                ) / (tf.math.reduce_std(videos_squeeze, axis=-2, keepdims=True) + 1e-6)
            videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
        else:
            videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)

        B, _, C = videos_patch.shape
        labels = tf.boolean_mask(videos_patch, bool_masked_pos).numpy().reshape(B, -1, C)

        with tf.GradientTape() as tape:
            outputs = model([videos, bool_masked_pos], training=True)
            loss = loss_func(labels, outputs)

        loss_value = loss.numpy()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        gradients = tape.gradient(loss, model.trainable_variables)
        if max_norm > 0:
            gradients = [tf.clip_by_norm(g, max_norm) for g in gradients]
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        if hasattr(optimizer, 'lr'):
            max_lr = optimizer.lr.numpy()
            min_lr = optimizer.lr.numpy()

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = optimizer.weight_decay if hasattr(optimizer, 'weight_decay') else None
        metric_logger.update(weight_decay=weight_decay_value)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
