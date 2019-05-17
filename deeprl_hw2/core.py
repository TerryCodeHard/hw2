"""Core classes."""
from collections import deque
import random, time
import warnings
import numpy as np


from keras.callbacks import History



class Sample:
    """Represents a reinforcement learning sample.

    Used to store observed experience from an MDP. Represents a
    standard `(s, a, r, s', terminal)` tuple.

    Note: This is not the most efficient way to store things in the
    replay memory, but it is a convenient class to work with when
    sampling batches, or saving and loading samples while debugging.

    Parameters
    ----------
    state: array-like
      Represents the state of the MDP before taking an action. In most
      cases this will be a numpy array.
    action: int, float, tuple
      For discrete action domains this will be an integer. For
      continuous action domains this will be a floating point
      number. For a parameterized action MDP this will be a tuple
      containing the action and its associated parameters.
    reward: float
      The reward received for executing the given action in the given
      state and transitioning to the resulting state.
    next_state: array-like
      This is the state the agent transitions to after executing the
      `action` in `state`. Expected to be the same type/dimensions as
      the state.
    is_terminal: boolean
      True if this action finished the episode. False otherwise.
    """

    def __init__(self, state, action, reward, next_state, is_terminal):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.is_terminal = is_terminal




class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v

    def clear(self):
        self.length = 0
        self.data = [None for _ in range(self.maxlen)]
        self.start = 0


# sample indexes by given low and high index, also the number of index we want
def sample_batch_indexes(low, high, size):
    if high - low >= size:
        r = range(low, high)

        batch_idxs = random.sample(r, size)
    else:
        # when data is not enough, we oversample
        warnings.warn('Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!')
        batch_idxs = np.random.random_integers(low, high - 1, size=size)
    assert len(batch_idxs) == size
    return batch_idxs


# this is for padding observation with all zero
def zeroed_observation(observation):
    if hasattr(observation, 'shape'):
        return np.zeros(observation.shape)
    else:
        assert 1 == 0, 'The input doesn''t have shape property'
class Preprocessor:
    """Preprocessor base class.

    This is a suggested interface for the preprocessing steps. You may
    implement any of these functions. Feel free to add or change the
    interface to suit your needs.

    Preprocessor can be used to perform some fixed operations on the
    raw state from an environment. For example, in ConvNet based
    networks which use image as the raw state, it is often useful to
    convert the image to greyscale or downsample the image.

    Preprocessors are implemented as class so that they can have
    internal state. This can be useful for things like the
    AtariPreproccessor which maxes over k frames.

    If you're using internal states, such as for keeping a sequence of
    inputs like in Atari, you should probably call reset when a new
    episode begins so that state doesn't leak in from episode to
    episode.
    """

    def process_state_for_network(self, state):
        """Preprocess the given state before giving it to the network.

        Should be called just before the action is selected.

        This is a different method from the process_state_for_memory
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory is a lot more efficient thant float32, but the
        networks work better with floating point images.

        Parameters
        ----------
        state: np.ndarray
          Generally a numpy array. A single state from an environment.

        Returns
        -------
        processed_state: np.ndarray
          Generally a numpy array. The state after processing. Can be
          modified in anyway.

        """

        return state

    def process_state_for_memory(self, state):
        """Preprocess the given state before giving it to the replay memory.

        Should be called just before appending this to the replay memory.

        This is a different method from the process_state_for_network
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory and the network expecting images in floating
        point.

        Parameters
        ----------
        state: np.ndarray
          A single state from an environmnet. Generally a numpy array.

        Returns
        -------
        processed_state: np.ndarray
          Generally a numpy array. The state after processing. Can be
          modified in any manner.

        """
        return state

    def process_batch(self, samples):
        """Process batch of samples.

        If your replay memory storage format is different than your
        network input, you may want to apply this function to your
        sampled batch before running it through your update function.

        Parameters
        ----------
        samples: list(tensorflow_rl.core.Sample)
          List of samples to process

        Returns
        -------
        processed_samples: list(tensorflow_rl.core.Sample)
          Samples after processing. Can be modified in anyways, but
          the list length will generally stay the same.
        """
        return samples

    def process_reward(self, reward):
        """Process the reward.

        Useful for things like reward clipping. The Atari environments
        from DQN paper do this. Instead of taking real score, they
        take the sign of the delta of the score.

        Parameters
        ----------
        reward: float
          Reward to process

        Returns
        -------
        processed_reward: float
          The processed reward
        """
        return reward

    def reset(self):
        """Reset any internal state.

        Will be called at the start of every new episode. Makes it
        possible to do history snapshots.
        """
        pass


class ReplayMemory:
    """Interface for replay memories.

    We have found this to be a useful interface for the replay
    memory. Feel free to add, modify or delete methods/attributes to
    this class.

    It is expected that the replay memory has implemented the
    __iter__, __getitem__, and __len__ methods.

    If you are storing raw Sample objects in your memory, then you may
    not need the end_episode method, and you may want to tweak the
    append method. This will make the sample method easy to implement
    (just ranomly draw saamples saved in your memory).

    However, the above approach will waste a lot of memory (as states
    will be stored multiple times in s as next state and then s' as
    state, etc.). Depending on your machine resources you may want to
    implement a version that stores samples in a more memory efficient
    manner.

    Methods
    -------
    append(state, action, reward, debug_info=None)
      Add a sample to the replay memory. The sample can be any python
      object, but it is suggested that tensorflow_rl.core.Sample be
      used.
    end_episode(final_state, is_terminal, debug_info=None)
      Set the final state of an episode and mark whether it was a true
      terminal state (i.e. the env returned is_terminal=True), of it
      is is an artificial terminal state (i.e. agent quit the episode
      early, but agent could have kept running episode).
    sample(batch_size, indexes=None)
      Return list of samples from the memory. Each class will
      implement a different method of choosing the
      samples. Optionally, specify the sample indexes manually.
    clear()
      Reset the memory. Deletes all references to the samples.
    """
    def __init__(self, max_size, input_shape, window_length, datatype='uint8'):
        """Setup memory.

        You should specify the maximum size o the memory. Once the
        memory fills up oldest values should be removed. You can try
        the collections.deque class as the underlying storage, but
        your sample method will be very slow.

        We recommend using a list as a ring buffer. Just track the
        index where the next sample should be inserted in the list.
        """
        self.max_size = int(max_size)
        self.window_length = window_length
        self.input_shape = input_shape
        self.datatype = datatype
        self.actions = RingBuffer(self.max_size)
        self.rewards = RingBuffer(self.max_size)
        self.terminals = RingBuffer(self.max_size)
        self.observations = RingBuffer(self.max_size)
        pass

    def append(self, observation, action, reward, terminal):
        assert observation.shape == self.input_shape
        assert observation.dtype == self.datatype
        self.observations.append(observation)  # 84x84
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)

    def end_episode(self, final_state, is_terminal):
        raise NotImplementedError('This method should be overridden')

    def sample(self, batch_size, indexes=None):
        if indexes is None:
            # Draw random indexes such that we have at least a single entry before each index.
            indexes = sample_batch_indexes(0, len(self) - 1, size=batch_size)
        indexes = np.array(indexes) + 1
        assert np.min(indexes) >= 1
        assert np.max(indexes) < len(self)
        assert len(indexes) == batch_size

        # Create experiences
        experiences = []
        for idx in indexes:
            terminal0 = self.terminals[idx - 2] if idx >= 2 else False
            while terminal0:
                # Skip this transition because there is terminal inside this tuple
                idx = sample_batch_indexes(1, len(self), size=1)[0]
                terminal0 = self.terminals[idx - 2] if idx >= 2 else False
            assert 1 <= idx < len(self)

            # create experience with specific window length by reversing the index of tuple in the memory
            state0 = np.empty([self.input_shape[0], self.input_shape[1], self.window_length], dtype=self.datatype)
            state1 = np.empty([self.input_shape[0], self.input_shape[1], self.window_length], dtype=self.datatype)
            state0[:, :, self.window_length - 1] = self.observations[idx - 1]

            cur_len = 1
            for offset in range(0, self.window_length - 1):  # offset is 0-2
                current_idx = idx - 2 - offset
                current_terminal = self.terminals[current_idx - 1] if current_idx - 1 > 0 else False
                if current_idx < 0 or current_terminal:
                    break
                state0[:, :, self.window_length - 2 - offset] = self.observations[current_idx]
                cur_len += 1
            while cur_len < self.window_length:
                # if the experience doesn't have enough previous observation inside the same episode, we pad zero observation
                state0[:, :, self.window_length - 1 - cur_len] = zeroed_observation(
                    state0[:, :, self.window_length - 1])
                cur_len += 1
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]

            # create the experience for next state by shifting the state by one
            state1_tmp = np.array([np.copy(x) for x in state0[:, :, 1:]])
            state1[:, :, 0:self.window_length - 1] = state1_tmp
            state1[:, :, self.window_length - 1] = self.observations[idx]  # add to the tail
            assert cur_len == self.window_length
            assert state1.shape == state0.shape

            experiences.append(Sample(state0, action, reward, state1, terminal1))
        assert len(experiences) == batch_size
        return experiences

    def clear(self):
        self.actions.clear()
        self.observations.clear()
        self.rewards.clear()
        self.terminals.clear()

    def get_config(self):
        config = {
            'window_length': self.window_length,
            'max_size': self.max_size,
        }
        return config

    @property
    def nb_entries(self):
        return len(self)

    def __len__(self):
        assert len(self.observations) == len(self.actions), 'The length of observation and action \
           in memory should be equal'
        assert len(self.actions) == len(self.rewards), 'The length of rewards and action \
           in memory should be equal'
        assert len(self.rewards) == len(self.terminals), 'The length of rewards and terminals \
           in memory should be equal'
        return len(self.observations)

    def print_tuple(self, index):
        print('state is: ')
        print(self.observations[index])
        print('action is: ')
        print(self.actions[index])
        print('reward is: ')
        print(self.rewards[index])
        print('is_terminal: ')
        print(self.terminals[index])

