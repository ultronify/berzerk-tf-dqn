import os
import gym
import random
import numpy as np
import tensorflow as tf
import logging
from tqdm import tqdm
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, BatchNormalization


def build_q_net(input_shape, action_space_size, learning_rate, game_id):
    print('Initialize Q net with action space size {0} and state shape {1}'.format(
        action_space_size, input_shape))
    model = Sequential()
    if game_id in ('Berzerk-v0', 'Skiing-v0', 'Assault-v0', 'Breakout-v0'):
        model.add(Conv2D(16, (4, 4), strides=(2, 2), padding='same',
                         kernel_initializer='he_uniform',
                         activation='relu', input_shape=input_shape))
        model.add(Conv2D(32, (4, 4), strides=(2, 2),
                         padding='same', kernel_initializer='he_uniform',
                         activation='relu'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2),
                         padding='same', kernel_initializer='he_uniform',
                         activation='relu'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(512, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(256, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(128, activation='relu',
                        kernel_initializer='he_uniform'))
    elif game_id in ('CartPole-v1', 'Breakout-ram-v0'):
        model.add(Dense(512, input_shape=input_shape,
                        activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(256, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(128, activation='relu',
                        kernel_initializer='he_uniform'))
    else:
        raise RuntimeError('Model for {0} not found'.format(game_id))
    model.add(Dense(action_space_size, activation='linear'))
    model.compile(optimizer=tf.optimizers.Adam(
        learning_rate=learning_rate), loss='mse')
    return model


def sanitize_state(state, game_id):
    if game_id in ('Berzerk-v0', 'Breakout-v0'):
        sanitized_state = np.array(state)
        sanitized_state = sanitized_state[0:-28, :, :]
        sanitized_state = np.expand_dims(
            np.dot(sanitized_state, [0.2989, 0.5870, 0.1140]), axis=2)
        sanitized_state = sanitized_state / 255.0
        return sanitized_state
    else:
        return state


def sample_action(q_vals, epsilon, action_space, collect=False):
    if np.random.random() < epsilon and collect:
        return np.random.randint(0, action_space)
    else:
        return np.argmax(q_vals)


def eval(q_net, max_eps, env, action_space_size, max_steps, render, game_id, state_buffer_size):
    total_reward = 0.0
    total_normalized_reward = 0.0
    state_buffer = deque(maxlen=state_buffer_size)
    for _ in range(max_eps):
        done = False
        state = env.reset()
        sanitized_initial_state = sanitize_state(state, game_id)
        for _ in range(state_buffer_size):
            state_buffer.append(sanitized_initial_state)
        step = 0
        while not done and step < max_steps:
            if render:
                env.render()
            frames = np.swapaxes(list(state_buffer), 0, 3)
            q_vals = q_net(tf.convert_to_tensor(
                frames, dtype=tf.float32))
            action = sample_action(
                q_vals, 0.0, action_space_size, collect=False)
            state, raw_reward, done, _ = env.step(action)
            state_buffer.append(sanitize_state(state, game_id))
            reward = normalize_reward(raw_reward, game_id)
            total_reward += raw_reward
            total_normalized_reward += reward
            step += 1
    avg_reward = total_reward / max_eps
    avg_normalized_reward = total_normalized_reward / max_eps
    return avg_reward, avg_normalized_reward


def normalize_reward(reward, game_id):
    if game_id == 'Berzerk-v0':
        if reward > 0:
            return 1.0
        elif reward < 0:
            return -5.0
        else:
            return -0.001
    elif game_id == 'CartPole-v1':
        return reward if reward > 0 else -1.0
    elif game_id == 'Skiing-v0':
        if reward > 0:
            return 1.0
        elif reward < 0:
            return -0.1
        else:
            return 0.0
    elif game_id == 'Breakout-v0':
        if reward > 0:
            return 1.0
        elif reward < 0:
            return -1.0
        else:
            return 0.0
    elif game_id in ('Breakout-ram-v0', 'Breakout-v0'):
        if reward > 0:
            return 1.0
        elif reward < 0:
            return -5.0
        else:
            return 0.0
    else:
        return reward


def explore(game_id='Berzerk-v0'):
    logging.info('Exploring {0}'.format(game_id))
    env = gym.make(game_id)
    print('Envrionment action space {0}, state space {1}'.format(
        env.action_space, env.observation_space))
    env.reset()
    rewards = []
    for _ in range(5):
        done = False
        while not done:
            env.render()
            _, reward, done, _ = env.step(env.action_space.sample())
            rewards.append(reward)
    rewards_np = np.array(rewards)
    print('Reward history size {0}, mean {1}, range {2} to {3}'.format(
        len(rewards), rewards_np.mean(), rewards_np.min(), rewards_np.max()))
    env.close()


def train(game_id='Berzerk-v0',
          max_eps=10000,
          max_buffer_size=5000,
          batch_size=512,
          gamma=0.95,
          max_eval_eps=3,
          update_freq=3,
          eval_freq=10,
          epsilon=0.8,
          epsilon_decay=0.99,
          max_steps=2000,
          render=True,
          checkpoint_location='most_recent',
          model_location='most_recent',
          save_checkpoint=True,
          learning_rate=1e-3,
          train_per_episode=1,
          rounds_per_episode=1,
          state_buffer_size=4,
          mode='train'):
    logging.info('Training {0}'.format(game_id))
    env = gym.make(game_id)
    eval_env = gym.make(game_id)
    use_epsilon = epsilon
    if render:
        env.render()
        if mode == 'train':
            eval_env.render()
    best_avg_reward = 0.0
    action_space_size = 0
    action_space_size = env.action_space.n
    state_buffer = deque(maxlen=state_buffer_size)
    sanitized_initial_state = sanitize_state(env.reset(), game_id)
    logging.info('Sanitized state shape is {0}'.format(sanitized_initial_state.shape))
    for _ in range(state_buffer_size):
        state_buffer.append(sanitized_initial_state)
    state_shape = np.swapaxes(np.array(list(state_buffer)), 0, 3).shape[1:]
    q_net = build_q_net(state_shape, action_space_size, learning_rate, game_id)
    target_q_net = build_q_net(
        state_shape, action_space_size, learning_rate, game_id)
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), net=q_net)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, os.path.join('./checkpoint', checkpoint_location), max_to_keep=10)
    try:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        target_q_net.set_weights(q_net.get_weights())
    except:
        q_net = build_q_net(state_shape, action_space_size,
                            learning_rate, game_id)
        target_q_net = build_q_net(
            state_shape, action_space_size, learning_rate, game_id)
    q_net.summary()
    replay_buffer = deque(maxlen=max_buffer_size)
    for eps in range(max_eps):
        # Sampling
        if mode in ('train', 'eval'):
            for _ in range(rounds_per_episode):
                done = False
                state = env.reset()
                for _ in range(state_buffer_size):
                    state_buffer.append(sanitized_initial_state)
                step = 0
                while not done and step < max_steps:
                    if render:
                        env.render()
                    frames = np.swapaxes(list(state_buffer), 0, 3)
                    q_vals = q_net(tf.convert_to_tensor(
                        frames, dtype=tf.float32))
                    action = sample_action(
                        q_vals, use_epsilon, action_space_size, collect=(mode == 'train'))
                    next_state, raw_reward, done, _ = env.step(action)
                    reward = normalize_reward(raw_reward, game_id)
                    state_buffer.append(sanitize_state(next_state, game_id))
                    updated_frames = np.swapaxes(list(state_buffer), 0, 3)
                    next_q_vals = target_q_net(tf.convert_to_tensor(updated_frames, dtype=tf.float32)).numpy()[0]
                    update_q_val = q_vals.numpy()[0]
                    update_q_val[action] = reward
                    if not done:
                        update_q_val[action] += gamma * next_q_vals.max()
                    replay_buffer.append(
                        (frames[0], updated_frames[0], reward, done, action, update_q_val))
                    state = next_state
                    step += 1
        # Training
        if mode == 'train':
            for _ in range(train_per_episode):
                # Prepare batch for training
                sampled_batch = []
                if len(replay_buffer) < batch_size:
                    sampled_batch = list(replay_buffer)
                else:
                    sampled_batch = random.sample(replay_buffer, batch_size)
                # actual_batch_size = len(sampled_batch)
                logging.info('Preparing batch...')
                states, next_states, rewards, actions, terminals, update_q_vals = [], [], [], [], [], []
                for state, next_state, reward, done, action, update_q_val in tqdm(sampled_batch):
                    states.append(state)
                    next_states.append(next_state)
                    rewards.append([reward])
                    terminals.append([0.0 if done else 1.0])
                    actions.append(action)
                    update_q_vals.append(update_q_val)
                """
                logging.info('Calculating loss...')
                target_q_vals = q_net(tf.convert_to_tensor(
                    states, dtype=tf.float32))
                next_q_vals = target_q_net(tf.convert_to_tensor(
                    next_states, dtype=tf.float32))
                max_next_q_vals = tf.expand_dims(
                    tf.reduce_max(next_q_vals, axis=1), axis=1)
                action_onehot = tf.one_hot(actions, action_space_size)
                action_onehot_reverse = tf.ones_like(
                    action_onehot) - action_onehot
                exclude_update_q_vals = target_q_vals * action_onehot_reverse
                update_q_vals_reward = action_onehot * \
                    tf.convert_to_tensor(rewards)
                update_q_vals_discounted_max_q = action_onehot * \
                    max_next_q_vals * gamma * tf.convert_to_tensor(terminals)
                target_q_vals = exclude_update_q_vals + \
                    update_q_vals_discounted_max_q + update_q_vals_reward
                logging.info('Start training...')
                q_net.fit(x=tf.convert_to_tensor(states, dtype=tf.float32),
                          y=tf.convert_to_tensor(
                              target_q_vals, dtype=tf.float32),
                          batch_size=16, verbose=1)
                """
                q_net.fit(x=tf.convert_to_tensor(states, dtype=tf.float32),
                          y=tf.convert_to_tensor(
                              update_q_vals, dtype=tf.float32),
                          batch_size=16, verbose=1)
                if save_checkpoint:
                    checkpoint_manager.save()
        if eps % update_freq == 0 and eps != 0 and mode == 'train':
            target_q_net.set_weights(q_net.get_weights())
        if eps % eval_freq == 0 and eps != 0 and mode == 'train':
            avg_reward, avg_normalized_reward = eval(q_net, max_eval_eps, eval_env,
                                                     action_space_size, max_steps, render, game_id, state_buffer_size)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                tf.saved_model.save(q_net, os.path.join(
                    './best_model', model_location))
            print('Current eval average reward is {0} and normalized reward is {1}'.format(
                avg_reward, avg_normalized_reward))
        use_epsilon *= epsilon_decay
        print('Finished episode {0}/{1} and the next epsilon is {2}'.format(eps, max_eps, use_epsilon))
    env.close()
    eval_env.close()
