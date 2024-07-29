import os
import numpy as np
import math
import sys
from typing import Iterable, Optional
import tensorflow as tf
from tensorflow import keras
from keras import mixed_precision
from scipy.special import softmax


from mixup import Mixup
import utils

def train_class_batch(model, samples, target, criterion):
    outputs = model(samples, training=True)
    loss = criterion(target, outputs)
    return loss, outputs

def train_one_epoch(model, criterion, data_loader, optimizer, epoch, mixup_fn=None, log_writer=None,
                    lr_schedule_values=None, wd_schedule_values=None, num_training_steps_per_epoch=None, update_freq=1):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 10

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = epoch * num_training_steps_per_epoch + step  # global training iteration

        if lr_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = tf.convert_to_tensor(samples)
        targets = tf.convert_to_tensor(targets)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with tf.GradientTape() as tape:
            loss, output = train_class_batch(model, samples, targets, criterion)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loss_value = loss.numpy()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        if mixup_fn is None:
            class_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis=-1), tf.argmax(targets, axis=-1)), tf.float32)).numpy()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.set_step()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@tf.function
def validation_one_epoch(data_loader, model):
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        videos = tf.convert_to_tensor(videos)
        target = tf.convert_to_tensor(target)

        output = model(videos, training=False)
        loss = criterion(target, output)

        acc1 = tf.keras.metrics.sparse_top_k_categorical_accuracy(target, output, k=1)
        acc5 = tf.keras.metrics.sparse_top_k_categorical_accuracy(target, output, k=5)

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.numpy())
        metric_logger.meters['acc1'].update(acc1.numpy(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.numpy(), n=batch_size)
    
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@tf.function
def final_test(data_loader, model, file):
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()
    final_result = []
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        videos = tf.convert_to_tensor(videos)
        target = tf.convert_to_tensor(target)

        output = model(videos, training=False)
        loss = criterion(target, output)

        for i in range(output.shape[0]):
            string = "{} {} {} {} {}\n".format(ids[i], \
                                                str(output.numpy()[i].tolist()), \
                                                str(int(target.numpy()[i])), \
                                                str(int(chunk_nb.numpy()[i])), \
                                                str(int(split_nb.numpy()[i])))
            final_result.append(string)

        acc1 = tf.keras.metrics.sparse_top_k_categorical_accuracy(target, output, k=1)
        acc5 = tf.keras.metrics.sparse_top_k_categorical_accuracy(target, output, k=5)

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.numpy())
        metric_logger.meters['acc1'].update(acc1.numpy(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.numpy(), n=batch_size)

    if not os.path.exists(file):
        open(file, 'w').close()
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]

def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float, sep=',')
            data = softmax(data)
            if name not in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1 ,final_top5 = np.mean(top1), np.mean(top5)
    return final_top1*100 ,final_top5*100