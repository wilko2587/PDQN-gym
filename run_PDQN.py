from PDQN import PDQNAgent, play
import gym
import gym_platform
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

if __name__=='__main__':
    env = gym.make("Platform-v0")

    # Network/setup params
    actorNet_kwargs = {'hidden_layers': (128,), 'l2': 0, 'lr': 1e-3}
    paramNet_kwargs = {'hidden_layers': (128,), 'l2': 0, 'lr': 1e-4}
    Nepisodes = 20000

    # initialise PDQN agent
    agent = PDQNAgent(observation_space=env.observation_space,
                          action_space=env.action_space,
                          actorNet_kwargs=actorNet_kwargs,
                          paramNet_kwargs=paramNet_kwargs,
                          train_start=500,
                          epsilon_decay=0.9995,
                          epsilon_min=0.01,
                          epsilon_bumps=[], # can reset epsilon to init value when it hits values inside this list
                          memory_size=10000,
                          batch_size=128,
                          gamma=0.9,
                          grad_clipping=2.,
                          stratify_replay_memory=False)

    # train agent, and get scores for each episode
    scores = play(env, agent, episodes=Nepisodes, render=True, train=True)

    agent.save(id=1)

    # ----- Plotting -----
    # bin the episodes into 500 length bins
    scores_binned = pd.DataFrame(index=np.floor(np.arange(0, len(scores)) / 500.) * 500, columns=['score'], data=scores)
    scores_binned = scores_binned.reset_index()
    scores_binned = scores_binned.rename(columns={'index': 'episode'})
    f = plt.figure()
    sns.pointplot(data=scores_binned, y='score', x='episode', errwidth=0.5, linewidth=0.5)

    # save results
    plt.savefig('result.png')
    scores_binned.to_csv('results.csv')
