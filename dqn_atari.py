#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random

import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Conv2D, Dense, Flatten, Input,
                          Permute, Reshape)
from keras.models import Model
from keras.optimizers import Adam
from keras.models import Sequential


import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
import keras

def create_model(window, input_shape, num_actions, type,
                 model_name='q_network'):  # noqa: D103
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance (you can also
    use the SequentialModel class).

    We highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understnad your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """

    if (type == 'deep'):

        with tf.name_scope(model_name):
            model = Sequential()
            model.add(Conv2D(filters = 32, kernel_size = 8, strides = (4,4), input_shape=(input_shape[0], input_shape[1], window), name='conv1'))
            model.add(Activation('relu', name='relu1'))
            model.add(Conv2D(filters = 64, kernel_size = 4, strides = (2,2), name='conv2'))
            model.add(Activation('relu', name='relu2'))
            model.add(Conv2D(filters = 64, kernel_size = 3, strides = (1,1), name='conv3'))
            model.add(Activation('relu', name='relu3'))
            model.add(Flatten())
            model.add(Dense(512, name='dense1'))
            model.add(Activation('relu', name='relu4'))
            model.add(Dense(num_actions, name='dense2'))
            model.add(Activation('linear', name='linear1'))

    elif (type == 'linear'):
        with tf.name_scope(model_name):
            model = Sequential()
            model.add(Reshape((input_shape[0]*input_shape[1]*window, ), input_shape=(input_shape[0], input_shape[1], window), name='reshape1'))
            model.add(Dense(num_actions, name='dense1'))
            model.add(Activation('linear', name='linear1'))

    print(model.summary())

    return model


def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--env', default='Breakout-v0', help='Atari env name')
    parser.add_argument(
        '-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    args = parser.parse_args()
    args.input_shape = tuple(args.input_shape)

    args.output = get_output_folder(args.output, args.env)

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.

if __name__ == '__main__':
    main()
