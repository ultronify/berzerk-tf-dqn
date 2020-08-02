import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, BatchNormalization


def build_q_net(input_shape, action_space_size):
    print('Initialize Q net with action space size {0} and state shape {1}'.format(
        action_space_size, input_shape))
    model = Sequential()
    model.add(Conv2D(16, (8, 8), strides=(2, 2), padding='same',
                     kernel_initializer='he_uniform',
                     activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (4, 4), strides=(2, 2),
                     padding='same', kernel_initializer='he_uniform',
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(action_space_size, activation='linear'))
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3), loss='mse')
    return model


def sanitize_state(state):
    sanitized_state = np.array(state)
    sanitized_state = sanitized_state[0:-28, :, :]
    # I think color matters so revert greyscale conversion
    # sanitized_state = np.expand_dims(np.dot(sanitized_state, [0.2989, 0.5870, 0.1140]), axis=2)
    sanitized_state = sanitized_state / 255.0
    return sanitized_state


def sample_action(q_vals, epsilon, action_space, collect=False):
    if np.random.random() < epsilon and collect:
        return np.random.randint(0, action_space)
    else:
        return np.argmax(q_vals)


def eval(q_net, max_eps, env, action_space_size, max_steps, render):
    total_reward = 0.0
    total_normalized_reward = 0.0
    for _ in range(max_eps):
        done = False
        state = env.reset()
        step = 0
        while not done and step < max_steps:
            if render:
                env.render()
            sanitized_state = sanitize_state(state)
            q_vals = q_net(tf.convert_to_tensor(
                [sanitized_state], dtype=tf.float32))
            action = sample_action(
                q_vals, 0.0, action_space_size, collect=False)
            state, raw_reward, done, _ = env.step(action)
            reward = normalize_reward(raw_reward)
            total_reward += raw_reward
            total_normalized_reward += reward
            step += 1
    avg_reward = total_reward / max_eps
    avg_normalized_reward = total_normalized_reward / max_eps
    return avg_reward, avg_normalized_reward


def normalize_reward(reward):
    if reward > 0:
        return 1.0
    elif reward < 0:
        return -1.0
    else:
        return 0.0


def train(max_eps=10000, max_buffer_size=10000,
          batch_size=256, gamma=0.95, max_eval_eps=3,
          update_freq=3, eval_freq=10, epsilon=0.1,
          max_steps=2000, render=True, checkpoint_location='./checkpoint',
          save_checkpoint=True):
    env = gym.make('Berzerk-v0')
    eval_env = gym.make('Berzerk-v0')
    if render:
        env.render()
        eval_env.render()
    best_avg_reward = 0.0
    action_space_size = env.action_space.n
    state_shape = sanitize_state(env.reset()).shape
    q_net = build_q_net(state_shape, action_space_size)
    target_q_net = build_q_net(state_shape, action_space_size)
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), net=q_net)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, checkpoint_location, max_to_keep=10)
    try:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        target_q_net.set_weights(q_net.get_weights())
    except:
        q_net = build_q_net(state_shape, action_space_size)
        target_q_net = build_q_net(state_shape, action_space_size)
    q_net.summary()
    replay_buffer = deque(maxlen=max_buffer_size)
    for eps in range(max_eps):
        done = False
        state = env.reset()
        step = 0
        while not done and step < max_steps:
            if render:
                env.render()
            sanitized_state = sanitize_state(state)
            q_vals = q_net(tf.convert_to_tensor(
                [sanitized_state], dtype=tf.float32))
            action = sample_action(
                q_vals, epsilon, action_space_size, collect=True)
            next_state, raw_reward, done, _ = env.step(action)
            reward = normalize_reward(raw_reward)
            replay_buffer.append(
                (sanitized_state, sanitize_state(next_state), reward, done, action))
            state = next_state
            step += 1
        # Prepare batch for training
        sampled_batch = []
        if len(replay_buffer) < batch_size:
            sampled_batch = list(replay_buffer)
        else:
            sampled_batch = random.sample(replay_buffer, batch_size)
        actual_batch_size = len(sampled_batch)
        states, next_states, rewards, actions, terminals = [], [], [], [], []
        for state, next_state, reward, done, action in sampled_batch:
            states.append(state)
            next_states.append(next_state)
            rewards.append(reward)
            terminals.append(done)
            actions.append(action)
        target_q_vals = q_net(tf.convert_to_tensor(
            states, dtype=tf.float32)).numpy()
        next_q_vals = target_q_net(tf.convert_to_tensor(
            next_states, dtype=tf.float32)).numpy()
        for i in range(actual_batch_size):
            if terminals[i]:
                target_q_vals[i, actions[i]] = rewards[i]
            else:
                target_q_vals[i, actions[i]] = rewards[i] + \
                    gamma * next_q_vals[i, :].max()
        q_net.fit(x=tf.convert_to_tensor(states, dtype=tf.float32),
                  y=tf.convert_to_tensor(target_q_vals, dtype=tf.float32),
                  batch_size=16, verbose=1)
        checkpoint_manager.save()
        if eps % update_freq == 0 and eps != 0:
            target_q_net.set_weights(q_net.get_weights())
        if eps % eval_freq == 0 and eps != 0:
            avg_reward, avg_normalized_reward = eval(q_net, max_eval_eps, eval_env,
                                                     action_space_size, max_steps, render)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                tf.saved_model.save(q_net, './best_model')
            print('Current eval average reward is {0} and normalized reward is {1}'.format(
                avg_reward, avg_normalized_reward))
        print('Finished episode {0}/{1}'.format(eps, max_eps))
    env.close()
    eval_env.close()
