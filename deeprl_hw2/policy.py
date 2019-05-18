
"""RL Policy classes.
We have provided you with a base policy class, some example
implementations and some unimplemented classes that should be useful
in your code.
"""
import numpy as np
import attr


class Policy(object):
    """Base class representing an MDP policy.
    Policies are used by the agent to choose actions.
    Policies are designed to be stacked to get interesting behaviors
    of choices. For instances in a discrete action space the lowest
    level policy may take in Q-Values and select the action index
    corresponding to the largest value. If this policy is wrapped in
    an epsilon greedy policy then with some probability epsilon, a
    random action will be chosen.
    """
    def _set_agent(self, agent):
        self.agent = agent


    @property
    def metrics_names(self):
        return []

    @property
    def metrics(self):
        return []


    def select_action(self, **kwargs):
        """Used by agents to select actions.
        Returns
        -------
        Any:
          An object representing the chosen action. Type depends on
          the hierarchy of policy instances.
        """
        raise NotImplementedError('This method should be overriden.')


    def get_config(self):
        return {}


class UniformRandomPolicy(Policy):
    """Chooses a discrete action with uniform random probability.
    This is provided as a reference on how to use the policy class.
    Parameters
    ----------
    num_actions: int
      Number of actions to choose from. Must be > 0.
    Raises
    ------
    ValueError:
      If num_actions <= 0
    """

    def __init__(self, num_actions):
        assert num_actions >= 1
        self.num_actions = num_actions

    def select_action(self, **kwargs):
        """Return a random action index.
        This policy cannot contain others (as they would just be ignored).
        Returns
        -------
        int:
          Action index in range [0, num_actions)
        """
        return np.random.randint(0, self.num_actions)

    def get_config(self):
        return {'num_actions': self.num_actions}


class GreedyPolicy(Policy):
    """Always returns best action according to Q-values.
    This is a pure exploitation policy.
    """

    def select_action(self, q_values, **kwargs):

        if q_values.ndim == 2 and q_values.shape[0] == 1:
            q_values = q_values[0, :]

        assert q_values.ndim == 1
        return np.argmax(q_values)


class GreedyEpsilonPolicy(Policy):
    """Selects greedy action or with some probability a random action.
    Standard greedy-epsilon implementation. With probability epsilon
    choose a random action. Otherwise choose the greedy action.
    Parameters
    ----------
    epsilon: float
     Initial probability of choosing a random action. Can be changed
     over time.
    """
    def __init__(self, epsilon):
        super(GreedyEpsilonPolicy, self).__init__()
        self.eps = epsilon

    def select_action(self, q_values, **kwargs):
        """Run Greedy-Epsilon for the given Q-values.
        Parameters
        ----------
        q_values: array-like
          Array-like structure of floats representing the Q-values for
          each action.
        Returns
        -------
        int:
          The action index chosen.
        """

        # take the q_value from 2d array to a vector
        if q_values.ndim == 2 and q_values.shape[0] == 1:
            q_values = q_values[0, :]
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:  # if epsilon, random
            action = np.random.random_integers(0, nb_actions-1)
        else:                               # if not, argmax
            action = np.argmax(q_values)
        return action



class LinearDecayGreedyEpsilonPolicy(Policy):
    """Policy with a parameter that decays linearly.
    Like GreedyEpsilonPolicy but the epsilon decays from a start value
    to an end value over k steps.
    Parameters
    ----------
    start_value: int, float
      The initial value of the parameter
    end_value: int, float
      The value of the policy at the end of the decay.
    num_steps: int
      The number of steps over which to decay the value.
    """

    def __init__(self, policy, attr_name, start_value, end_value, num_steps):
    #    if not hasattr(policy, attr_name):
      #      raise ValueError('Policy "{}" does not have attribute "{}".'.format(attr_name))
        super(LinearDecayGreedyEpsilonPolicy, self).__init__()

        self.policy = policy
        self.attr_name = attr_name
        self.end_value = end_value
        self.start_value = start_value
        self.num_steps = num_steps

    def get_current_value(self, is_training=True):
        if is_training:
            a = -float(self.start_value - self.end_value) / float(self.num_steps)
            b = float(self.start_value)
            value = max(self.end_value, a * float(self.agent.step) + b)     # decay the episilon based on current training step
        else:
            value = self.end_value
        return value

    def select_action(self, is_training=True, **kwargs):
        """Decay parameter and select action.
        Parameters
        ----------
        q_values: np.array
          The Q-values for each action.
        is_training: bool, optional
          If true then parameter will be decayed. Defaults to true.
        Returns
        -------
        Any:
          Selected action.
        """
        setattr(self.policy, self.attr_name, self.get_current_value(is_training))
        return self.policy.select_action(**kwargs)


    @property
    def metrics_names(self):
        return ['{}'.format(self.attr_name)]

    # return the current episilon for monitoring
    @property
    def metrics(self):
        return [getattr(self.policy, self.attr_name)]


    def get_config(self):
        config = super(LinearDecayGreedyEpsilonPolicy, self).get_config()
        config['attr_name'] = self.attr
        config['start_value'] = self.start_value
        config['end_value'] = self.end_value
        config['num_steps'] = self.num_steps
        config['policy'] = get_object_config(self.policy)
        return config

