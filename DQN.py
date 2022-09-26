from datetime import datetime
import random
import gym
import gym_platform
import numpy as np
import pandas as pd
from collections import deque
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt


# policy network
def OurModel(input_shape, action_space):
    X_input = Input(input_shape)

    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)

    # Hidden layer with 256 nodes
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)

    # Hidden layer with 64 nodes
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    # Output Layer with # of actions: 2 nodes (left, right)
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs=X_input, outputs=X, name='CartPole_DQN_model')
    model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    # model.summary()
    return model


class Agent:
    def __init__(self):

        self.state_size = 9
        self.action_size = 3 #self.env.action_space.n
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.000
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 1000

        # create main model
        self.model = OurModel(input_shape=(self.state_size,), action_space=self.action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        # implement the epsilon-greedy policy
        if len(state.shape) == 1:  # catch for dimensionality issues
            state = np.array(state).reshape([1, len(state)])
        q_out = self.model.predict(state, verbose=0)
        action = np.argmax(q_out)

        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(range(self.action_size))
        else:
            return action

    def replay(self):
        if len(self.memory) < self.train_start:
            return

        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # assign data into state, next_state, action, reward and done from minibatch
        for i in range(self.batch_size):
            ministate, miniaction, minireward, mininextstate, minidone = minibatch[i]
            state[i] = ministate
            next_state[i] = mininextstate
            action.append(miniaction)
            reward.append(minireward)
            done.append(minidone)

        target = np.zeros((self.batch_size, self.action_size))
        # compute value function of current(call it target) and value function of next state(call it target_next)
        for i in range(self.batch_size):
            # correction on the Q value for the action used,
            # if done[i] is true, then the target should be just the final reward
            a = action[i]
            if done[i]:
                # print(state[i], next_state[i], action[i])
                target[i] = self.model.predict(state[i].reshape([1, len(state[i])]), verbose=0)
                target[i, a] = reward[i]  # -100
            else:
                Q_target_next = np.max(self.model.predict(next_state[i].reshape([1, len(next_state[i])]), verbose=0))
                target[i] = self.model.predict(state[i].reshape([1, len(state[i])]), verbose=0)
                target[i, a] = reward[i] + self.gamma * Q_target_next

        self.model.fit(state, target, batch_size=self.batch_size, verbose=False, epochs=1)

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)


def train(env, agent, episodes=10):

    scores = []
    for e in range(episodes):
        state, _ = env.reset()
        print('state: {}'.format(state))
        state = np.transpose(state)
        done = False
        i = 0

        actions = []
        while not done:
            # self.env.render()

            action = agent.act(state)
            print('    action: {}'.format(action))
            actions.append(action)
            next_state, reward, done, _ = env.step([action])

            print('    new state: {}'.format(next_state))
            next_state = np.transpose(next_state)
            if not done or i == env._max_episode_steps - 1:
                reward = reward
            else:
                reward = -100

            agent.remember(state, action, reward, next_state, done)

            state = next_state
            i += 1
            if done:
                dateTimeObj = datetime.now()
                timestampStr = dateTimeObj.strftime("%H:%M:%S")
                print("episode: {}/{}, score: {}, e: {:.2}, time: {}".format(e + 1, self.EPISODES, i, self.epsilon,
                                                                             timestampStr))
                scores.append(i)

            agent.replay()
    return scores

if __name__ == "__main__":

    env = gym.make("Platform-v0")
    agent = Agent()
    scores = train(env, agent)

    #plt.figure()
    #plt.plot(scores, label='example learning curve')
    #plt.legend()
    #plt.xlabel('episode')
    #plt.ylabel('score')

    #plt.show()
