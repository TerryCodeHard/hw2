"""Suggested Preprocessors."""

import numpy as np
from PIL import Image
from collections import deque

from deeprl_hw2 import utils
from deeprl_hw2.core import Preprocessor


class HistoryPreprocessor(Preprocessor):
    """Keeps the last k states.

    Useful for domains where you need velocities, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Parameters
    ----------
    history_length: int
      Number of previous states to prepend to state being processed.

    """

    def __init__(self, history_length=1):
        self.history_length = history_length
        self.history = deque([], maxlen=history_length)
        return

    def process_state_for_network(self, state):
        """You only want history when you're deciding the current action to take."""
        assert state.dtype == 'uint8'
        assert state.shape == (84, 84)

        if (len(self.history) < self.history_length - 1):
            for i in range(len(self.history), self.history_length - 1):
                self.history.append(np.zeros(state.shape, dtype='uint8'))

        assert len(self.history) >= self.history_length - 1

        self.history.append(state)
        history_state = np.zeros((1, state.shape[0], state.shape[1], self.history_length), dtype='uint8')
        for i in range(len(self.history)):
            history_state[:, :, :, i] = self.history[i]

        return history_state

    def reset(self):
        """Reset the history sequence.

        Useful when you start a new episode.
        """
        self.history.clear()
        return

    def get_config(self):
        return {'history_length': self.history_length}


class AtariPreprocessor(Preprocessor):
    """Converts images to greyscale and downscales.

    Based on the preprocessing step described in:

    @article{mnih15_human_level_contr_throug_deep_reinf_learn,
    author =	 {Volodymyr Mnih and Koray Kavukcuoglu and David
                  Silver and Andrei A. Rusu and Joel Veness and Marc
                  G. Bellemare and Alex Graves and Martin Riedmiller
                  and Andreas K. Fidjeland and Georg Ostrovski and
                  Stig Petersen and Charles Beattie and Amir Sadik and
                  Ioannis Antonoglou and Helen King and Dharshan
                  Kumaran and Daan Wierstra and Shane Legg and Demis
                  Hassabis},
    title =	 {Human-Level Control Through Deep Reinforcement
                  Learning},
    journal =	 {Nature},
    volume =	 518,
    number =	 7540,
    pages =	 {529-533},
    year =	 2015,
    doi =        {10.1038/nature14236},
    url =	 {http://dx.doi.org/10.1038/nature14236},
    }

    You may also want to max over frames to remove flickering. Some
    games require this (based on animations and the limited sprite
    drawing capabilities of the original Atari).

    Parameters
    ----------
    new_size: 2 element tuple
      The size that each image in the state should be scaled to. e.g
      (84, 84) will make each image in the output have shape (84, 84).
    """

    def __init__(self, new_size):
        self.new_size = new_size
        pass

    def process_state_for_memory(self, state):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """

        assert state.ndim == 3  # (height, width, channel)
        image_tmp = Image.fromarray(state)
        state_unit8 = np.asarray(image_tmp.resize(self.new_size).convert('L'))
        state_unit8.astype('uint8')
        assert state_unit8.shape == self.new_size

        return state_unit8


    def process_state_for_network(self, state):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        return self.process_state_for_memory(state).astype('float32') / 255.


    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        samples.state = self.process_state_for_memory(samples.state).astype('float32') / 255.
        samples.next_state = self.process_state_for_memory(samples.next_state).astype('float32') / 255.
        return samples

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        return np.clip(reward, -1.0, 1.0)


class PreprocessorSequence(Preprocessor):
    """You may find it useful to stack multiple prepcrocesosrs (such as the History and the AtariPreprocessor).

    You can easily do this by just having a class that calls each preprocessor in succession.

    For example, if you call the process_state_for_network and you
    have a sequence of AtariPreproccessor followed by
    HistoryPreprocessor. This this class could implement a
    process_state_for_network that does something like the following:

    state = atari.process_state_for_network(state)
    return history.process_state_for_network(state)
    """

    def __init__(self, preprocessors):
        self.Atari = preprocessors['Atari']
        self.History = preprocessors['History']

    def process_state_for_network(self, observation):
        '''
        observation: 84x84 uint8
        return: 84x84 float32
        '''

        assert observation.dtype == 'uint8', 'observation in forward is not correct'
        assert observation.shape == (84, 84)
        tmp = self.History.process_state_for_network(observation)
        processed_state = tmp.astype('float32') / 255.

        random_index = np.random.randint(84, size=1)[0]
        # assert processed_state.shape == (1, 84, 84, 4)
        assert processed_state[0, random_index, random_index, 0] <= 1. and processed_state[
                                                                               0, random_index, random_index, 0] >= 0., 'processed state is not correct while forward'
        return processed_state

    def process_state_from_memory_batch(self, batch_state_from_memory):
        """The batches from replay memory will be uint8, convert to float32.
        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        batch_state_from_memory is a list which has length of batch_size, each item
        is a state with shape 84x84x4
        return a numpy array with shape 32x84x84x4
        """
        batch_num = len(batch_state_from_memory)
        random_batch = np.random.randint(batch_num, size=1)[0]

        assert batch_state_from_memory[random_batch].shape == (84, 84, 4)
        assert batch_state_from_memory[random_batch].dtype == 'uint8'
        batch_state_processed = np.array(batch_state_from_memory).astype('float32') / 255.

        # assert batch_state_processed.shape == (batch_num, 84, 84, 4)
        random_index = np.random.randint(84, size=1)[0]
        assert batch_state_processed[random_batch, random_index, random_index, 0] >= 0 and batch_state_processed[
                                                                                               random_batch, random_index, random_index, 0] <= 1

        return batch_state_processed
