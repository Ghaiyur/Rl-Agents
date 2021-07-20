# Imports
import random
import numpy as np
import flappy_bird_gym
from collections import deque
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import RMSprop
import tensorflow

from keras.models import save_model
from keras.models import load_model

# Neural Network For Agent


def NN(input_shape, output_shape):
    model = Sequential()
    model.add(Dense(512, input_shape=input_shape,
              activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(256,  activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(64,  activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(output_shape,  activation='linear',
              kernel_initializer='he_uniform'))
    model.compile(loss='mse', optimizer=RMSprop(
        learning_rate=0.0001, rho=0.95, epsilon=0.01), metrics=['accuracy'])
    # model.summary()
    return model


# Brain of Agent || BluePrint of Agent


class DQNAgent:
    def __init__(self):
        # Env Variables
        self.env = flappy_bird_gym.make('FlappyBird-v0')
        self.episodes = 1000
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.memory = deque(maxlen=2000)

        # Hyperparameter
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        self.batch_number = 256

        self.train_start = 257
        self.jump_prob = 0.01
        self.model = NN(input_shape=(self.state_space,),
                        output_shape=self.action_space)

    def act(self, state):
        if np.random.random() > self.epsilon:
            return np.argmax(self.model.predict(state))
        return 1 if np.random.random() < self.jump_prob else 0

    def learn(self):
        # Make sure we have enough data
        if len(self.memory) < self.train_start:
            return
        # Create minibatch
        minibatch = random.sample(self.memory, min(
            len(self.memory), self.batch_number))
        # Var to store minibatch info
        state = np.zeros((self.batch_number, self.state_space))
        next_state = np.zeros((self.batch_number, self.state_space))
        action, reward, done = [], [], []

        # Store Data in Var
        for i in range(self.batch_number):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # Predict y label
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.batch_number):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + \
                    self.gamma * (np.amax(target_next[i]))

        self.model.fit(state, target, batch_size=self.batch_number,
                       verbose=0)

    def get_config(self):
        cfg = super().get_config()
        return cfg

    def train(self):
        # n episodes iters
        for i in range(self.episodes):
            # Env Variables for training
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_space])
            done = False
            score = 0
            self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon * \
                self.epsilon_decay > self.epsilon_min else self.epsilon_min

            while not done:
                self.env.render()
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)

                # reshape nextstate
                next_state = np.reshape(next_state, [1, self.state_space])
                score += 1

                if done:
                    reward -= 100

                self.memory.append((state, action, reward, next_state, done))
                state = next_state

                if done:
                    print('Episode: {}\nScore: {}\nEpsilon: {}'.format(
                        i, score, self.epsilon))
                    # Save Model
                    if score >= 7000:
                        self.model.save(
                            'my_model.h5', save_format='h5', overwrite=True)
                        return

                self.learn()

    def perform(self):
        self.model = load_model('my_model.h5')
        while 1:
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_space])
            done = False
            score = 0

            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, info = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_space])
                score += 1

                print('Current Score: {}'.format(
                    score))

                if done:
                    print('DEAD')
                    break


if __name__ == '__main__':
    env = flappy_bird_gym.make('FlappyBird-v0')
    agent = DQNAgent()
    agent.train()
    agent.perform()
