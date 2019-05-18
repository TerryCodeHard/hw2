#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os, time
import random
import gym
from gym import wrappers

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential
from keras.layers import (Activation, Conv2D, Dense, Flatten, Input,
                          Permute, Reshape)
from keras.models import Model
from keras.optimizers import Adam, RMSprop
import keras

import deeprl_hw2.__init__
from deeprl_hw2.preprocessors import AtariPreprocessor, HistoryPreprocessor, PreprocessorSequence
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.policy import LinearDecayGreedyEpsilonPolicy, GreedyEpsilonPolicy, GreedyPolicy, Policy
from deeprl_hw2.core import ReplayMemory
from deeprl_hw2.callbacks import *


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
            print(model.summary())

            return model
    elif (type == 'linear'):
        with tf.name_scope(model_name):
            model = Sequential()
            model.add(Reshape((input_shape[0]*input_shape[1]*window, ), input_shape=(input_shape[0], input_shape[1], window), name='reshape1'))
            model.add(Dense(num_actions, name='dense1'))
            model.add(Activation('linear', name='linear1'))

            print(model.summary())

            return model


def get_session():
    num_threads = os.environ.get('OMP_NUM_THREADS')

    if num_threads:
        config = tf.ConfigProto(intra_op_parallelism_threads=num_threads)
        config.gpu_options.allow_growth = True
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

    return tf.Session(config=config)
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
    #os.makedirs(parent_dir)
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

def main():
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--env', default='Breakout-v0', help='Atari env name')
    parser.add_argument('-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--network', choices=['deep', 'linear'], default='deep')
    parser.add_argument('--method', choices=['dqn', 'double', 'dueling'], default='dqn')
    parser.add_argument('--monitor', type=bool, default=False)
    parser.add_argument('--iter', type=int, default=2400000)
    parser.add_argument('--test_policy', choices=['Greedy', 'GreedyEpsilon'], default='GreedyEpsilon')

    args = parser.parse_args()
    args.seed = np.random.randint(0, 1000000, 1)[0]
    args.weights = 'models/dqn_{}_weights_{}_{}_{}.h5f'.format(args.env, args.method, args.network, args.iter)
    args.monitor_path = 'tmp/dqn_{}_weights_{}_{}_{}_{}'.format(args.env, args.method, args.network, args.iter,
                                                                args.test_policy)
    if args.mode == 'train':
        args.monitor = False

    env = gym.make(args.env)
    if args.monitor:
        env = wrappers.Monitor(env, args.monitor_path, force = True)
    np.random.seed(args.seed)
    env.seed(args.seed)

    args.gamma = 0.99
    args.learning_rate = 0.0001
    args.epsilon = 0.05
    args.num_iterations = 5000000
    args.batch_size = 32

    args.window_length = 4
    args.num_burn_in = 50000
    args.target_update_freq = 10000
    args.log_interval = 10000
    args.model_checkpoint_interval = 10000
    args.train_freq = 4

    args.num_actions = env.action_space.n
    args.input_shape = (84, 84)
    args.memory_max_size = 1000000

    args.output = get_output_folder(args.output, args.env)

    args.suffix = args.method + '_' + args.network
    if (args.method == 'dqn'):
        args.enable_double_dqn = False
        args.enable_dueling_network = False
    elif (args.method == 'double'):
        args.enable_double_dqn = True
        args.enable_dueling_network = False
    elif (args.method == 'dueling'):
        args.enable_double_dqn = False
        args.enable_dueling_network = True
    else:
        print('Attention! Method Worng!!!')

    if args.test_policy == 'Greedy':
        test_policy = GreedyPolicy()
    elif args.test_policy == 'GreedyEpsilon':
        test_policy = GreedyEpsilonPolicy(args.epsilon)

    print(args)

    K.tensorflow_backend.set_session(get_session())
    model = create_model(args.window_length, args.input_shape, args.num_actions, args.network)

    # we create our preprocessor, the Ataripreprocessor will only process current frame the agent is seeing. And the sequence
    # preprocessor will construct the state by concatenating 3 previous frames from HistoryPreprocessor and current processed frame
    Processor = {}
    Processor['Atari'] = AtariPreprocessor(args.input_shape)
    Processor['History'] = HistoryPreprocessor(args.window_length)
    ProcessorSequence = PreprocessorSequence(Processor)  # construct 84x84x4

    # we create our memory for saving all experience collected during training with window length 4
    memory = ReplayMemory(max_size=args.memory_max_size, input_shape=args.input_shape, window_length=args.window_length)

    # we use linear decay greedy epsilon policy and tune the epsilon from 1 to 0.1 during the first 100w iterations and then keep using
    # epsilon with 0.1 to further train the network


    # we construct our agent and use 0.99 as our discounted factor, 32 as our batch_size. We update our model for each 4 iterations. But during first
    # 50000 iterations, we only collect data to the memory and don't update our model.
    policy = LinearDecayGreedyEpsilonPolicy(GreedyEpsilonPolicy(args.epsilon), attr_name='eps', start_value=1, end_value=0.1, num_steps=1000000)

    dqn = DQNAgent(q_network=model, policy=policy, memory=memory, num_actions=args.num_actions, test_policy=test_policy,
                   preprocessor=ProcessorSequence, gamma=args.gamma, target_update_freq=args.target_update_freq,
                   num_burn_in=args.num_burn_in, train_freq=args.train_freq, batch_size=args.batch_size,
                   enable_double_dqn=args.enable_double_dqn, enable_dueling_network=args.enable_dueling_network)

    adam = Adam(lr=args.learning_rate)
    dqn.compile(optimizer=adam)

    if args.mode == 'train':
        weights_filename = 'dqn_{}_weights_{}.h5f'.format(args.env, args.suffix)
        checkpoint_weights_filename = 'dqn_' + args.env + '_weights_' + args.suffix + '_{step}.h5f'
        log_filename = 'dqn_{}_log_{}.json'.format(args.env, args.suffix)
        log_dir = '../tensorboard_{}_log_{}'.format(args.env, args.suffix)
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=args.model_checkpoint_interval)]
        callbacks += [FileLogger(log_filename, interval=100)]
        callbacks += [TensorboardStepVisualization(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)]

        # start training
        # we don't apply action repetition explicitly since the game will randomly skip frame itself
        dqn.fit(env, callbacks=callbacks, verbose=1, num_iterations=args.num_iterations, action_repetition=1,
                log_interval=args.log_interval, visualize=True)

        dqn.save_weights(weights_filename, overwrite=True)
        dqn.evaluate(env, num_episodes=10, visualize=True, num_burn_in=5, action_repetition=1)
    elif args.mode == 'test':
        weights_filename = 'dqn_{}_weights_{}.h5f'.format(args.env, args.suffix)
        if args.weights:
            weights_filename = args.weights
        dqn.load_weights(weights_filename)
        dqn.evaluate(env, num_episodes=250, visualize=True, num_burn_in=5, action_repetition=1)

        # we upload our result to openai gym
        if args.monitor:
            env.close()


if __name__ == '__main__':
    main()