from utils import *
from objectives import huber_loss
from keras.layers import Lambda, Input, Layer, Dense
import keras.backend as K
from callbacks import TestLogger, TrainEpisodeLogger, TrainIntervalLogger, Visualizer, CallbackList
from keras.callbacks import History
from policy import *
from keras.utils import plot_model
import keras


import warnings
from copy import deepcopy
import time
import numpy as np
"""Main DQN agent."""

class DQNAgent(object):
    """Class implementing DQN.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    """
    def __init__(self,
                 q_network,
                 preprocessor,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size,
                 num_actions,
                 test_policy=None, 
                 enable_double_dqn=False,
                 enable_dueling_network=False,
                 max_grad=1.,
                 nb_max_start_steps=0):
        self.preprocessor = preprocessor
        self.memory = memory
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_network = enable_dueling_network
        self.num_actions = num_actions
        self.max_grad = max_grad
        self.dueling_type = 'avg'

        self.training = False
        self.step = 0

        if self.enable_dueling_network:
            # get the second last layer of the q_network, abandon the last layer
            layer = q_network.layers[-2]
            nb_action = q_network.output._keras_shape[-1]
            y = Dense(nb_action + 1, activation='linear')(layer.output)
            if self.dueling_type == 'avg':
                outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True), output_shape=(nb_action,))(y)
            elif self.dueling_type == 'max':
                outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True), output_shape=(nb_action,))(y)
            elif self.dueling_type == 'naive':
                outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(nb_action,))(y)
            else:
                assert False, "dueling_type must be one of {'avg','max','naive'}"
            q_network = Model(input=q_network.input, output=outputlayer)


        self.q_network = q_network
        if policy is None:
            policy = GreedyEpsilonPolicy()
        if test_policy is None:
            # test_policy = GreedyEpsilonPolicy(0.05)    # during testing, we use episilon greedy policy
            test_policy = GreedyPolicy()                 # during testing, we use greedy policy
        
        self.test_policy = test_policy
        self.__policy = policy

    def get_config(self):
        config = super(DQNAgent, self).get_config()
        config['enable_double_dqn'] = self.enable_double_dqn
        config['dueling_type'] = self.dueling_type
        config['enable_dueling_network'] = self.enable_dueling_network
        config['q_network'] = get_object_config(self.q_network)
        config['policy'] = get_object_config(self.policy)
        config['test_policy'] = get_object_config(self.test_policy)
        if self.compiled:
            config['target_model'] = get_object_config(self.target_model)
        return config


    @property
    def metrics_names(self):
        # Throw away individual losses and replace output name since this is hidden from the user.
        assert len(self.trainable_model.output_names) == 2
        dummy_output_name = self.trainable_model.output_names[1]
        model_metrics = [name for idx, name in enumerate(self.trainable_model.metrics_names) if idx not in (1, 2)]
        model_metrics = [name.replace(dummy_output_name + '_', '') for name in model_metrics]
        names = model_metrics + self.policy.metrics_names[:]
        return names

    @property
    def policy(self):
        self.__policy._set_agent(self)  # set the agent while calling the policy
        return self.__policy

    def reset_states(self):
        if self.compiled:               # reset the state
            self.q_network.reset_states()
            self.target_model.reset_states()

    def compile(self, optimizer, metrics=[]):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """

        metrics += [mean_q]  # register default combined_metrics
        
        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        self.target_model = clone_model(self.q_network)
        self.target_model.compile(optimizer=optimizer, loss='mse')
        self.q_network.compile(optimizer=optimizer, loss='mse')

        # construct a new loss for trainable_model, which uses mask to hide all action-q value 
        # except the one from target_network
        def clipped_masked_error(args):        
            y_pred, y_true, mask = args
            loss = huber_loss(y_true=y_true, y_pred=y_pred, max_grad=self.max_grad)
            loss *= mask  
            return K.sum(loss, axis=-1)
        y_pred = self.q_network.output
        y_true = Input(name='y_true', shape=(self.num_actions,))
        mask = Input(name='mask', shape=(self.num_actions,))
        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_pred, y_true, mask])

        # construct the trainable model by connecting original q_network with the new loss
        ins = [self.q_network.input] if type(self.q_network.input) is not list else self.q_network.input
        trainable_model = Model(input=ins + [y_true, mask], output=[loss_out, y_pred])
        assert len(trainable_model.output_names) == 2
        combined_metrics = {trainable_model.output_names[1]: metrics}     
        losses = [
            lambda y_true, y_pred: y_pred,                      # output the loss_out
            lambda y_true, y_pred: K.zeros_like(y_pred),        # output a dummy loss which is not used for back propagation
        ]
        trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
        self.trainable_model = trainable_model
        self.compiled = True


    def update_target_model_hard(self):
        self.target_model.set_weights(self.q_network.get_weights())


    def load_weights(self, filepath):
        self.q_network.load_weights(filepath)
        self.update_target_model_hard()


    def save_weights(self, filepath, overwrite=False):
        self.q_network.save_weights(filepath, overwrite=overwrite)


    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        assert state.dtype == 'float32'
        q_value = self.q_network.predict_on_batch(state)
        return q_value

    def select_action(self, observation, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """

        # from one observation (84x84 uint8) to get a state containing four 84x84x4 by calling
        # history preprocessor
        assert observation.shape == (84, 84)
        assert observation.dtype == 'uint8'
        state = self.preprocessor.process_state_for_network(observation)      
        q_values = self.calc_q_values(state)

        if self.training:
            action = self.policy.select_action(q_values=q_values)
        else:
            action = self.test_policy.select_action(q_values=q_values)

        return action

    def update_policy(self):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """
        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:   # if not training, just output the metrics
            return metrics

        # check if it's time to train
        if self.step % self.train_freq == 0 and self.step > self.num_burn_in:
            experiences = self.memory.sample(self.batch_size)   # get sample from memory
            assert len(experiences) == self.batch_size
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state)
                state1_batch.append(e.next_state)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.is_terminal else 1.)

            # convert sample from memory to state for network to train
            state0_batch = self.preprocessor.process_state_from_memory_batch(state0_batch)  
            state1_batch = self.preprocessor.process_state_from_memory_batch(state1_batch)
            assert state0_batch.dtype == 'float32'
            assert state1_batch.dtype == 'float32'

            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert len(action_batch) == len(reward_batch)

            # Compute Q values for mini-batch update.
            if self.enable_double_dqn:
                # get argmax action from q_network
                q_values = self.q_network.predict_on_batch(state1_batch)
                assert q_values.shape == (self.batch_size, self.num_actions)
                actions = np.argmax(q_values, axis=1)
                assert actions.shape == (self.batch_size,)

                # get target q value from target_network based on the best action computed from q_network
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.num_actions)
                q_batch = target_q_values[range(self.batch_size), actions]
            else:
                # compute max q value based on current state from target network as groundtruth
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.num_actions)
                q_batch = np.max(target_q_values, axis=1).flatten()
            assert q_batch.shape == (self.batch_size,)

            targets = np.zeros((self.batch_size, self.num_actions))
            dummy_targets = np.zeros((self.batch_size,))
            masks = np.zeros((self.batch_size, self.num_actions))
            discounted_reward_batch = self.gamma * q_batch
            discounted_reward_batch *= terminal1_batch      # mask the reward by terminal signal
            assert discounted_reward_batch.shape == reward_batch.shape
            Rs = reward_batch + discounted_reward_batch     # total reward computed from TD(0) equation
            for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
                target[action] = R                          # update action with estimated accumulated reward
                dummy_targets[idx] = R
                mask[action] = 1.                           # enable loss for this specific action
            targets = np.array(targets).astype('float32')
            masks = np.array(masks).astype('float32')

            # train the trainable model for the current batch
            ins = [state0_batch] if type(self.q_network.input) is not list else state0_batch
            metrics = self.trainable_model.train_on_batch(ins + [targets, masks], [dummy_targets, targets])
            metrics = [metric for idx, metric in enumerate(metrics) if idx not in (1, 2)]  # throw away intermediate losses
            metrics += self.policy.metrics                  # report current episilon

        # update current target model
        if self.step % self.target_update_freq == 0:
            self.update_target_model_hard()     

        return metrics


    def fit(self, env, callbacks, num_iterations, action_repetition=1, max_episode_length=None, log_interval=10000, verbose=1, visualize=False, validation_data=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """

        if not self.compiled:
            raise RuntimeError('Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')
        self.training = True
        self.validation_data = validation_data

        # for callback to record the log
        callbacks = [] if not callbacks else callbacks[:]
        if verbose == 1:
            callbacks += [TrainIntervalLogger(interval=log_interval)]
        elif verbose > 1:
            callbacks += [TrainEpisodeLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'num_iterations': num_iterations,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        callbacks.on_train_begin()


        # start training
        episode = 0
        self.step = 0
        observation = None
        episode_reward = None
        episode_step = None
        did_abort = False
        try:
            while (self.step < num_iterations):
                if observation is None:  # new episode
                    callbacks.on_episode_begin(episode)
                    episode_step = 0
                    episode_reward = 0.
                    self.reset_states()
                    observation = deepcopy(env.reset())
                    assert observation is not None

                assert episode_reward is not None
                assert episode_step is not None
                assert observation is not None

                # execute a new step
                callbacks.on_step_begin(episode_step)
                observation_tmp = self.preprocessor.Atari.process_state_for_memory(observation)  # cache the observation before action for saving to memory
                action = self.select_action(observation_tmp)        # run network forward to get a action

                # action repetition is for skipping frame by executing same action multiple times
                # but since we are using environment v0, no need to skip frame manually becaue random
                # skipping is executed by the environment itself
                reward = 0.
                accumulated_info = {}
                done = False
                for _ in range(action_repetition):      
                    callbacks.on_action_begin(action)
                    observation, r, done, info = env.step(action)
                    observation = deepcopy(observation)
                    r = self.preprocessor.Atari.process_reward(r)
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    callbacks.on_action_end(action)
                    reward += r
                    if done:
                        break

                if max_episode_length and episode_step >= max_episode_length - 1:
                    # Force a terminal state.
                    done = True

                # save the current tuple to memory
                self.memory.append(observation_tmp, action, reward, done)
                metrics = self.update_policy()
                weights = self.q_network.get_weights() 
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'metrics': metrics,
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)
                callbacks.pass_loss(self.step, step_logs)


                episode_step += 1
                self.step += 1

                if done:
                    observation_tmp = self.preprocessor.Atari.process_state_for_memory(observation)  # cache the observation before action for saving to memory
                    action = self.select_action(observation_tmp)    # one more step to a new episode
                    self.update_policy()
                    self.memory.append(observation_tmp, action, 0., False)  
                    self.preprocessor.History.reset()

                    # This episode is finished, report and reset.
                    episode_logs = {
                        'episode_reward': episode_reward,
                        'nb_episode_steps': episode_step,
                        'nb_steps': self.step,
                    }
                    callbacks.on_episode_end(episode, episode_logs)

                    episode += 1
                    observation = None
                    episode_step = None
                    episode_reward = None

        except KeyboardInterrupt:
            did_abort = True
        callbacks.on_train_end(logs={'did_abort': did_abort})
        self._on_train_end()

        return history





    def evaluate(self, env, num_episodes, action_repetition=1, max_episode_length=None, num_burn_in=10, callbacks=None, verbose=1, visualize=False):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        if not self.compiled:
            raise RuntimeError('Your tried to test your agent but it hasn\'t been compiled yet. Please call `compile()` before `test()`.')

        self.training = False
        self.step = 0

        callbacks = [] if not callbacks else callbacks[:]

        if verbose >= 1:
            callbacks += [TestLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'num_episodes': num_episodes,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)

        self._on_test_begin()
        callbacks.on_train_begin()
        for episode in range(num_episodes):
            # new episode
            callbacks.on_episode_begin(episode)
            episode_reward = 0.
            episode_step = 0

            # Obtain the initial observation by resetting the environment.
            self.reset_states()
            observation = deepcopy(env.reset())
            assert observation is not None

            # by using num_burn_in to slightly change the starting position at the beginning of the game
            print('Performing random action to change starting position at the beginning')
            for i in range(num_burn_in):
                action = env.action_space.sample()
                callbacks.on_action_begin(action)
                observation, reward, done, info = env.step(action)
                observation = deepcopy(observation)
                callbacks.on_action_end(action)
                if done:
                    warnings.warn('Env ended before {} random steps could be performed at the start. You should probably lower the `num_burn_in` parameter.'.format(num_burn_in))
                    observation = deepcopy(env.reset())
                    break

            # Run the episode until we're done.
            done = False
            while not done:
                callbacks.on_step_begin(episode_step)
                observation_tmp = self.preprocessor.Atari.process_state_for_memory(observation)  # cache the observation before action for saving to memory
                action = self.select_action(observation_tmp)
                # print(env.get_action_meanings()[action])

                # action repetition is for skipping frame by executing same action multiple times
                # but since we are using environment v0, no need to skip frame manually becaue random
                # skipping is executed by the environment itself
                reward = 0.
                accumulated_info = {}
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, d, info = env.step(action)
                    observation = deepcopy(observation)
                    callbacks.on_action_end(action)
                    reward += r
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    if d:
                        done = True
                        break
                if max_episode_length and episode_step >= max_episode_length - 1:
                    done = True

                episode_reward += reward        # reward is not clipped in the evaluation   
                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

            # Report end of episode.
            episode_logs = {
                'episode_reward': episode_reward,
                'nb_steps': episode_step,
            }
            callbacks.on_episode_end(episode, episode_logs)
        callbacks.on_train_end()
        self._on_test_end()

        return history


    def _on_test_begin(self):
        pass

    def _on_test_end(self):
        pass


    def _on_train_begin(self):
        pass

    def _on_train_end(self):
        pass