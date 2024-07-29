import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import argparse
import json
import os
from keras.optimizers import AdamW
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy


def create_model(args):
    inputs = layers.Input(shape=(args.num_frames, args.input_size, args.input_size, 3))
    x = layers.Conv3D(3, (args.tubelet_size, 1, 1), strides=(args.tubelet_size, 1, 1), padding='same')(inputs)
    x = layers.Reshape((x.shape[1], x.shape[2] * x.shape[3], x.shape[4]))(x)
    x = layers.Lambda(lambda x: tf.reshape(x, (-1, x.shape[2], x.shape[3])))(x)
    x = layers.Embedding(input_dim=args.input_size * args.input_size, output_dim=args.hidden_size, input_length=args.num_frames)(x)
    x = layers.Lambda(lambda x: tf.reshape(x, (-1, args.num_frames, args.hidden_size)))(x)
    x = layers.MultiHeadAttention(num_heads=args.num_heads, key_dim=args.hidden_size)(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(args.nb_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=x)
    return model


def train_one_epoch(model, criterion, optimizer, data_loader, epoch):
    model.trainable = True
    total_loss = 0
    for batch in data_loader:
        inputs, labels = batch
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            loss = criterion(labels, outputs)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        total_loss += loss
    return total_loss / len(data_loader)

def validation_one_epoch(model, data_loader):
    model.trainable = False
    total_correct = 0
    for batch in data_loader:
        inputs, labels = batch
        outputs = model(inputs, training=False)
        _, predicted = tf.nn.top_k(outputs, 1)
        total_correct += tf.reduce_sum(tf.cast(tf.equal(predicted, labels), tf.int32))
    return total_correct / len(data_loader)

def final_test(model, data_loader):
    model.trainable = False
    total_correct = 0
    for batch in data_loader:
        inputs, labels = batch
        outputs = model(inputs, training=False)
        _, predicted = tf.nn.top_k(outputs, 1)
        total_correct += tf.reduce_sum(tf.cast(tf.equal(predicted, labels), tf.int32))
    return total_correct / len(data_loader)




